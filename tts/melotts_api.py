import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["OMP_NUM_THREADS"] = "2"
import re
import time
import tempfile
import soundfile
import numpy as np
import onnxruntime as ort
import spacemit_ort
from .symbols import LANG_TO_SYMBOL_MAP
from .split_utils import split_sentence
from .text import cleaned_text_to_sequence


cur_dir = os.path.dirname(
    os.path.abspath(__file__)
)
lang = 'ZH_MIX_EN'
enc_model = os.path.join(cur_dir, 'models/encoder-zh.onnx')
dec_model = os.path.join(cur_dir, 'models/decoder-zh.onnx')


def intersperse(origin_items, item):
    result = [item] * (len(origin_items) * 2 + 1)
    result[1::2] = origin_items
    return result


def audio_numpy_concat(segment_data_list, sr, speed=1.):
    audio_segments = []
    for segment_data in segment_data_list:
        audio_segments += segment_data.reshape(-1).tolist()
        audio_segments += [0] * int((sr * 0.05) / speed)
    audio_segments = np.array(audio_segments).astype(np.float32)
    return audio_segments


def merge_sub_audio(sub_audio_list, pad_size, audio_len):
    # Average pad part
    if pad_size > 0:
        for i in range(len(sub_audio_list) - 1):
            sub_audio_list[i][-pad_size:] += sub_audio_list[i+1][:pad_size]
            sub_audio_list[i][-pad_size:] /= 2
            if i > 0:
                sub_audio_list[i] = sub_audio_list[i][pad_size:]

    sub_audio = np.concatenate(sub_audio_list, axis=-1)
    return sub_audio[:audio_len]


# 计算每个词的发音时长
def calc_word2pronoun(word2ph, pronoun_lens):
    indice = [0]
    for ph in word2ph[:-1]:
        indice.append(indice[-1] + ph)
    word2pronoun = []
    for i, ph in zip(indice, word2ph):
        word2pronoun.append(np.sum(pronoun_lens[i : i + ph]))
    return word2pronoun

# 生成有overlap的slice，slice索引是对于zp的
def generate_slices(word2pronoun, dec_len):
    pn_start, pn_end = 0, 0
    zp_start, zp_end = 0, 0
    zp_len = 0
    pn_slices = []
    zp_slices = []
    while pn_end < len(word2pronoun):
        # 前一个slice长度大于2 且 加上现在这个字没有超过dec_len，则往前overlap两个字
        if pn_end - pn_start > 2 and np.sum(word2pronoun[pn_end - 2 : pn_end + 1]) <= dec_len:
            zp_len = np.sum(word2pronoun[pn_end - 2 : pn_end])
            zp_start = zp_end - zp_len
            pn_start = pn_end - 2
        else:
            zp_len = 0
            zp_start = zp_end
            pn_start = pn_end
            
        while pn_end < len(word2pronoun) and zp_len + word2pronoun[pn_end] <= dec_len:
            zp_len += word2pronoun[pn_end]
            pn_end += 1
        zp_end = zp_start + zp_len
        pn_slices.append(slice(pn_start, pn_end))
        zp_slices.append(slice(zp_start, zp_end))
    return pn_slices, zp_slices


class MeloTTS:
    def __init__(self):
        self.lang = lang
        self._symbol_to_id = {
            s:i for i, s in enumerate(LANG_TO_SYMBOL_MAP[self.lang])
        }
        self._load_language_module()

        start = time.time()
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        self.sess_enc = ort.InferenceSession(
            enc_model,
            providers=['CPUExecutionProvider'],
            sess_options=sess_options
        )
        self.sess_dec = ort.InferenceSession(
            dec_model,
            providers=['SpaceMITExecutionProvider'],
            sess_options=sess_options
        )
        print(f"load models take {time.time() - start}s")
        self.dec_len = 65536 // 512
        self.sample_rate = 48000

        # Load static input
        self.g = np.fromfile(
            os.path.join(cur_dir, f'models/g-{self.lang.lower()}.bin'),
            dtype=np.float32
        ).reshape(1, 256, 1)

    def _load_language_module(self):
        start = time.time()
        if self.lang == "ZH_MIX_EN":
            from .text import chinese_mix as language_module
        elif self.lang == "EN":
            from .text import english as language_module
        else:
            assert False, f"Unsupported lanuguage: {self.lang}"
        print(f"Load language module take {time.time() - start}s")
        self.lang_module = language_module

    def _clean_text(self, text):
        norm_text = self.lang_module.text_normalize(text)
        phones, tones, word2ph = self.lang_module.g2p(norm_text)
        return norm_text, phones, tones, word2ph
    
    def _get_text_for_tts_infer(self, text):
        norm_text, phone, tone, word2ph = self._clean_text(text)
        phone, tone, language = cleaned_text_to_sequence(
            phone, tone, self.lang, self._symbol_to_id
        )

        phone = intersperse(phone, 0)
        tone = intersperse(tone, 0)
        language = intersperse(language, 0)

        phone = np.array(phone, dtype=np.int32)
        tone = np.array(tone, dtype=np.int32)
        language = np.array(language, dtype=np.int32)
        word2ph = np.array(word2ph, dtype=np.int32) * 2
        word2ph[0] += 1

        return phone, tone, language, norm_text, word2ph
    
    def _split_text_into_pieces(self, text, quiet=True):
        texts = split_sentence(text, language_str=self.lang)
        if not quiet:
            print(" > Text split to sentences.")
            print('\n'.join(texts))
            print(" > ===========================")
        return texts
    
    def run(self, text, speed=0.8):
        sentences = self._split_text_into_pieces(text)
        audio_list = []
        for n, sentence in enumerate(sentences):
            # print('origin:', sentence)
            if self.lang in ['EN', 'ZH_MIX_EN']:
                sentence = re.sub(r'([a-z])([A-Z])', r'\1 \2', sentence)
            # print(f'\nSentence[{n}]: {sentence}')

            # Convert sentence to phones and tones
            phones, tones, lang_ids, norm_text, word2ph = self._get_text_for_tts_infer(
                sentence
            )
            # print('phones:', len(phones), phones)
            # print('tones', len(tones), tones)
            # print('lang_ids:', lang_ids)
            # print('norm_text', norm_text)
            # print('word2ph', word2ph)

            start = time.time()
            # Run encoder
            z_p, pronoun_lens, audio_len = self.sess_enc.run(
                None,
                input_feed={
                    'phone': phones,
                    'g': self.g,
                    'tone': tones,
                    'language': lang_ids,
                    'noise_scale': np.array([0], dtype=np.float32),
                    'length_scale': np.array([1.0 / speed], dtype=np.float32),
                    'noise_scale_w': np.array([0], dtype=np.float32),
                    'sdp_ratio': np.array([0], dtype=np.float32)
                }
            )
            # print(f"encoder run take {1000 * (time.time() - start):.2f}ms")

            # 计算每个词的发音长度
            word2pronoun = calc_word2pronoun(word2ph, pronoun_lens)
            # 生成word2pronoun和zp的切片
            pn_slices, zp_slices = generate_slices(word2pronoun, self.dec_len)

            audio_len = audio_len[0]
            sub_audio_list = []
            for i, (ps, zs) in enumerate(zip(pn_slices, zp_slices)):
                zp_slice = z_p[..., zs]

                # Padding前zp的长度
                sub_dec_len = zp_slice.shape[-1]
                # Padding前输出音频的长度
                sub_audio_len = 512 * sub_dec_len

                # Padding到dec_len
                if zp_slice.shape[-1] < self.dec_len:
                    zp_slice = np.concatenate((zp_slice, np.zeros((*zp_slice.shape[:-1], self.dec_len - zp_slice.shape[-1]), dtype=np.float32)), axis=-1)

                # start = time.time()
                audio = self.sess_dec.run(
                    None,
                    input_feed={
                        "z_p": zp_slice,
                        "g": self.g
                    })[0].flatten()
                
                # 处理overlap
                audio_start = 0
                if len(sub_audio_list) > 0:
                    if pn_slices[i - 1].stop > ps.start:
                        # 去掉第一个字
                        audio_start = 512 * word2pronoun[ps.start]
        
                audio_end = sub_audio_len
                if i < len(pn_slices) - 1:
                    if ps.stop > pn_slices[i + 1].start:
                        # 去掉最后一个字
                        audio_end = sub_audio_len - 512 * word2pronoun[ps.stop - 1]

                audio = audio[audio_start:audio_end]
                # print(f"Decode slice[{i}]: decoder run take {1000 * (time.time() - start):.2f}ms")
                sub_audio_list.append(audio)
            sub_audio = merge_sub_audio(sub_audio_list, 0, audio_len)
            audio_list.append(sub_audio)
        audio = audio_numpy_concat(audio_list, sr=self.sample_rate, speed=speed)
        temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio_file = temp_wav_file.name
        soundfile.write(audio_file, audio, self.sample_rate)
        return audio_file
        

if __name__ == '__main__':
    text = "今天天气怎么样，你还好吗"
    melo = MeloTTS()
    print(melo.run(text))
    t1 = time.time()
    text = '讲一个白雪公主的故事'
    print(melo.run(text))
    print('used time:', time.time() - t1)
