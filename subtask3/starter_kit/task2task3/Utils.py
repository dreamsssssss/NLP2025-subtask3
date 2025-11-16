import os
import math
import torch
import logging
import numpy as np

from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

class ReviewDataset(Dataset):
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        dataset_to_numpy_array = {
            'line': self.dataset[item].line,

            'forward_asp_query': np.array(self.dataset[item].forward_asp_query),
            'forward_opi_query': np.array(self.dataset[item].forward_opi_query),
            'forward_asp_query_mask': np.array(self.dataset[item].forward_asp_query_mask),
            'forward_opi_query_mask': np.array(self.dataset[item].forward_opi_query_mask),
            'forward_asp_query_seg': np.array(self.dataset[item].forward_asp_query_seg),
            'forward_opi_query_seg': np.array(self.dataset[item].forward_opi_query_seg),
            'forward_asp_answer_start': np.array(self.dataset[item].forward_asp_answer_start),
            'forward_asp_answer_end': np.array(self.dataset[item].forward_asp_answer_end),
            'forward_opi_answer_start': np.array(self.dataset[item].forward_opi_answer_start),
            'forward_opi_answer_end': np.array(self.dataset[item].forward_opi_answer_end),

            'backward_asp_query': np.array(self.dataset[item].backward_asp_query),
            'backward_opi_query': np.array(self.dataset[item].backward_opi_query),
            'backward_asp_query_mask': np.array(self.dataset[item].backward_asp_query_mask),
            'backward_opi_query_mask': np.array(self.dataset[item].backward_opi_query_mask),
            'backward_asp_query_seg': np.array(self.dataset[item].backward_asp_query_seg),
            'backward_opi_query_seg': np.array(self.dataset[item].backward_opi_query_seg),
            'backward_asp_answer_start': np.array(self.dataset[item].backward_asp_answer_start),
            'backward_asp_answer_end': np.array(self.dataset[item].backward_asp_answer_end),
            'backward_opi_answer_start': np.array(self.dataset[item].backward_opi_answer_start),
            'backward_opi_answer_end': np.array(self.dataset[item].backward_opi_answer_end),

            # 'category_query': np.array(self.dataset[item].category_query),
            # 'category_answer': np.array(self.dataset[item].category_answer),
            # 'category_query_mask': np.array(self.dataset[item].category_query_mask),
            # 'category_query_seg': np.array(self.dataset[item].category_query_seg),

            'valence_query': np.array(self.dataset[item].valence_query),
            'valence_answer': np.array(self.dataset[item].valence_answer),
            'valence_query_mask': np.array(self.dataset[item].valence_query_mask),
            'valence_query_seg': np.array(self.dataset[item].valence_query_seg),

            'arousal_query': np.array(self.dataset[item].arousal_query),
            'arousal_answer': np.array(self.dataset[item].arousal_answer),
            'arousal_query_mask': np.array(self.dataset[item].arousal_query_mask),
            'arousal_query_seg': np.array(self.dataset[item].arousal_query_seg)
        }

        if self.args.task == 3 and None not in self.dataset[item].category_query:
            dataset_to_numpy_array['category_query'] = np.array(self.dataset[item].category_query)
            dataset_to_numpy_array['category_answer'] = np.array(self.dataset[item].category_answer)
            dataset_to_numpy_array['category_query_mask'] = np.array(self.dataset[item].category_query_mask)
            dataset_to_numpy_array['category_query_seg'] = np.array(self.dataset[item].category_query_seg)
        return dataset_to_numpy_array

    def get_batch_num(self, batch_size):
        if len(self.dataset) % batch_size == 0:
            return len(self.dataset) / batch_size
        return int(len(self.dataset) / batch_size) + 1


class InferenceReviewDataset(Dataset):
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        dataset_to_numpy_array = {
            'id': self.dataset[item].id,
            'line': self.dataset[item].line,
            'forward_asp_query': np.array(self.dataset[item].forward_asp_query),
            'forward_asp_query_mask': np.array(self.dataset[item].forward_asp_query_mask),
            'forward_asp_query_seg': np.array(self.dataset[item].forward_asp_query_seg),
            'forward_asp_answer_start': np.array(self.dataset[item].forward_asp_answer_start),
            'forward_asp_answer_end': np.array(self.dataset[item].forward_asp_answer_end),

            'backward_opi_query': np.array(self.dataset[item].backward_opi_query),
            'backward_opi_query_mask': np.array(self.dataset[item].backward_opi_query_mask),
            'backward_opi_query_seg': np.array(self.dataset[item].backward_opi_query_seg),
            'backward_opi_answer_start': np.array(self.dataset[item].backward_opi_answer_start),
            'backward_opi_answer_end': np.array(self.dataset[item].backward_opi_answer_end),
        }
        return dataset_to_numpy_array

    def get_batch_num(self, batch_size):
        if len(self.dataset) % batch_size == 0:
            return len(self.dataset) / batch_size
        return int(len(self.dataset) / batch_size) + 1


class QueryAndAnswer:
    def __init__(self, line, forward_asp_query, forward_opi_query,
                 forward_asp_query_mask, forward_asp_query_seg,
                 forward_opi_query_mask, forward_opi_query_seg,
                 forward_asp_answer_start, forward_asp_answer_end,
                 forward_opi_answer_start, forward_opi_answer_end,

                 backward_asp_query, backward_opi_query,
                 backward_asp_answer_start, backward_asp_answer_end,
                 backward_asp_query_mask, backward_asp_query_seg,
                 backward_opi_query_mask, backward_opi_query_seg,
                 backward_opi_answer_start, backward_opi_answer_end,

                 category_query, category_answer,
                 category_query_mask, category_query_seg,

                 valence_query, valence_answer,
                 valence_query_mask, valence_query_seg,

                 arousal_query, arousal_answer,
                 arousal_query_mask, arousal_query_seg,
                 ):
        self.line = line

        self.forward_asp_query = forward_asp_query
        self.forward_opi_query = forward_opi_query
        self.forward_asp_query_mask = forward_asp_query_mask
        self.forward_asp_query_seg = forward_asp_query_seg
        self.forward_opi_query_mask = forward_opi_query_mask
        self.forward_opi_query_seg = forward_opi_query_seg
        self.forward_asp_answer_start = forward_asp_answer_start
        self.forward_asp_answer_end = forward_asp_answer_end
        self.forward_opi_answer_start = forward_opi_answer_start
        self.forward_opi_answer_end = forward_opi_answer_end

        self.backward_asp_query = backward_asp_query
        self.backward_opi_query = backward_opi_query
        self.backward_asp_query_mask = backward_asp_query_mask
        self.backward_asp_query_seg = backward_asp_query_seg
        self.backward_opi_query_mask = backward_opi_query_mask
        self.backward_opi_query_seg = backward_opi_query_seg
        self.backward_asp_answer_start = backward_asp_answer_start
        self.backward_asp_answer_end = backward_asp_answer_end
        self.backward_opi_answer_start = backward_opi_answer_start
        self.backward_opi_answer_end = backward_opi_answer_end

        self.category_query = category_query
        self.category_answer = category_answer
        self.category_query_mask = category_query_mask
        self.category_query_seg = category_query_seg

        self.valence_query = valence_query
        self.valence_answer = valence_answer
        self.valence_query_mask = valence_query_mask
        self.valence_query_seg = valence_query_seg

        self.arousal_query = arousal_query
        self.arousal_answer = arousal_answer
        self.arousal_query_mask = arousal_query_mask
        self.arousal_query_seg = arousal_query_seg


class Query:
    def __init__(self, text_id, line, forward_asp_query,
                 forward_asp_query_mask, forward_asp_query_seg,
                 forward_asp_answer_start, forward_asp_answer_end,
                 backward_opi_query, backward_opi_query_mask, backward_opi_query_seg,
                 backward_opi_answer_start, backward_opi_answer_end
                 ):
        self.id = text_id
        self.line = line

        self.forward_asp_query = forward_asp_query
        self.forward_asp_query_mask = forward_asp_query_mask
        self.forward_asp_query_seg = forward_asp_query_seg
        self.forward_asp_answer_start = forward_asp_answer_start
        self.forward_asp_answer_end = forward_asp_answer_end
        self.backward_opi_query=backward_opi_query
        self.backward_opi_query_mask = backward_opi_query_mask
        self.backward_opi_query_seg = backward_opi_query_seg
        self.backward_opi_answer_start = backward_opi_answer_start
        self.backward_opi_answer_end = backward_opi_answer_end


class TestDataset:
    def __init__(self, line, aspect_list, opinion_list, asp_opi_list, asp_cate_list, triplet_list, valence_list,
                 arousal_list, VA_list):
        self.line = line
        self.aspect_list = aspect_list
        self.opinion_list = opinion_list
        self.asp_opi_list = asp_opi_list
        self.asp_cate_list = asp_cate_list
        self.triplet_list = triplet_list
        self.valence_list = valence_list
        self.arousal_list = arousal_list
        self.VA_list = VA_list


def normalize_size(tensor):
    if len(tensor.size()) == 3:
        tensor = tensor.contiguous().view(-1, tensor.size(2))
    elif len(tensor.size()) == 2:
        tensor = tensor.contiguous().view(-1)
    return tensor


def replace_using_dict(text, replacement_dict):
    for search, replace in replacement_dict.items():
        text = text.replace(search, replace)
    return text


def calculate_entity_loss(pred_start, pred_end, gold_start, gold_end, gpu):
    pred_start = normalize_size(pred_start)
    pred_end = normalize_size(pred_end)
    gold_start = normalize_size(gold_start)
    gold_end = normalize_size(gold_end)

    weight = torch.tensor([1, 3]).float()
    if gpu:
        weight = weight.cuda()
    loss_start = F.cross_entropy(pred_start, gold_start.long(), reduction='sum', weight=weight, ignore_index=-1)
    loss_end = F.cross_entropy(pred_end, gold_end.long(), reduction='sum', weight=weight, ignore_index=-1)
    return 0.5 * loss_start + 0.5 * loss_end


def calculate_category_loss(pred_category, gold_category):
    return F.cross_entropy(pred_category, gold_category.long(), reduction='sum', ignore_index=-1)


def calculate_valence_loss(pred_valence, gold_valence):
    # return F.cross_entropy(pred_valence, gold_valence.long(), reduction='sum', ignore_index=-1)
    return F.mse_loss(pred_valence, gold_valence.float(), reduction='sum')


def calculate_arousal_loss(pred_arousal, gold_arousal):
    # return F.cross_entropy(pred_arousal, gold_arousal.long(), reduction='sum', ignore_index=-1)
    return F.mse_loss(pred_arousal, gold_arousal.float(), reduction='sum')


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger, fh, sh


def filter_unpaired(start_prob, end_prob, start, end, max_len):
    filtered_start = []
    filtered_end = []
    filtered_prob = []
    if len(start) > 0 and len(end) > 0:
        length = start[-1] + 1 if start[-1] >= end[-1] else end[-1] + 1
        temp_seq = [0] * length
        for s in start:
            temp_seq[s] += 1
        for e in end:
            temp_seq[e] += 2
        start_index = []
        for idx in range(len(temp_seq)):
            assert temp_seq[idx] < 4
            if temp_seq[idx] == 1:
                start_index.append(idx)
            elif temp_seq[idx] == 2:
                if len(start_index) != 0 and (idx - start_index[-1] + 1) <= max_len:
                    max_prob = 0
                    max_prob_index = 0
                    for index in start_index:
                        if max_prob <= start_prob[start.index(index)] and \
                                (idx - index + 1) <= max_len:
                            max_prob = start_prob[start.index(index)]
                            max_prob_index = index
                    filtered_start.append(max_prob_index)
                    filtered_end.append(idx)
                    filtered_prob.append(math.sqrt(max_prob * end_prob[end.index(idx)]))
                start_index = []
            elif temp_seq[idx] == 3:
                start_index.append(idx)
                max_prob = 0
                max_prob_index = 0
                for index in start_index:
                    if max_prob <= start_prob[start.index(index)] and \
                            (idx - index + 1) <= max_len:
                        max_prob = start_prob[start.index(index)]
                        max_prob_index = index
                filtered_start.append(max_prob_index)
                filtered_end.append(idx)
                filtered_prob.append(math.sqrt(max_prob * end_prob[end.index(idx)]))
                start_index = []
    return filtered_start, filtered_end, filtered_prob


def _sanitize_for_collate(obj):
    """
    把 numpy 裡面 dtype 是 object / string 的陣列轉成 Python list，
    避免 default_collate 直接噴錯。
    """
    if isinstance(obj, np.ndarray):
        # kind: 'O' = object, 'U'/'S' = unicode/bytes 字串
        if obj.dtype.kind in {"O", "U", "S"}:
            return [_sanitize_for_collate(x) for x in obj.tolist()]
        else:
            return obj

    if isinstance(obj, dict):
        return {k: _sanitize_for_collate(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_collate(x) for x in obj]

    return obj


def _safe_collate(batch):
    """
    比原本的 default_collate 多做幾件事：
    1. 丟掉整個 sample 是 None 的情況
    2. 處理 dict 裡 value 是 None 的欄位，幫它用同型別的零值補起來
    3. 處理 numpy dtype=object -> 轉成正常的數值 array
    """
    # 1. 先把整個 sample 是 None 的丟掉
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        raise RuntimeError("All samples in this batch are None. Please check dataset preprocessing.")

    # 如果不是 dict，直接交給 default_collate（例如純 tensor 列表）
    if not isinstance(batch[0], dict):
        return default_collate(batch)

    # 2. 為每個 key 找一個「prototype」（第一個非 None 的值）
    prototypes = {}
    for sample in batch:
        if not isinstance(sample, dict):
            continue
        for k, v in sample.items():
            if v is None:
                continue
            if k not in prototypes:
                prototypes[k] = v

    cleaned_batch = []
    for sample in batch:
        fixed = {}
        for k, proto in prototypes.items():
            v = sample.get(k, None)

            # 處理 numpy object array
            if isinstance(v, np.ndarray) and v.dtype == object:
                v = np.array(list(v), dtype=np.int64)

            # 3. 如果這個 sample 的這個欄位是 None，用 prototype 建一個「零值」來補
            if v is None:
                p = proto
                if isinstance(p, torch.Tensor):
                    v = torch.zeros_like(p)
                elif isinstance(p, np.ndarray):
                    v = np.zeros_like(p)
                elif isinstance(p, (int, float)):
                    v = type(p)(0)
                elif isinstance(p, str):
                    v = ""
                elif isinstance(p, (list, tuple)):
                    v = type(p)([0] * len(p))
                else:
                    # 其他少見型別，直接複製 prototype 也行
                    v = p

            fixed[k] = v

        cleaned_batch.append(fixed)

    return default_collate(cleaned_batch)



def generate_batches(dataset, batch_size, shuffle=True, gpu=False):
    """
    包一層 DataLoader：
    - 用 _safe_collate 避開 dtype=object/str 的 numpy 陣列
    - 如果 gpu=True，會自動把 tensor 丟到 cuda
    """
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_safe_collate,
    )

    use_cuda = gpu and torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    for batch in dataloader:
        if isinstance(batch, dict):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
        yield batch



def create_directory(arguments):
    if not os.path.exists(arguments.log_path):
        os.makedirs(arguments.log_path)
    if not os.path.exists(arguments.save_model_path):
        os.makedirs(arguments.save_model_path)
    if not os.path.exists(arguments.output_path):
        subtask2_path = arguments.output_path + "subtask_2/"
        subtask3_path = arguments.output_path + "subtask_3/"
        os.makedirs(arguments.output_path)
        os.makedirs(subtask2_path)
        os.makedirs(subtask3_path)
    log_path = arguments.log_path + arguments.model_name + '.log'
    model_path = arguments.save_model_path + arguments.model_name + '.pth'

    if not os.path.exists(log_path):
        log = open(log_path, 'w')
        log.close()
        model = open(model_path, 'w')
        model.close()

def combine_lists(list1, list2):
    combinations = [f"{s1}#{s2}" for s1 in list1 for s2 in list2]
    result_dict = {}
    for index, combo in enumerate(combinations):
        result_dict[combo] = index
    return result_dict, combinations

def replace_using_dict(text, replacement_dict):
    for search, replace in replacement_dict.items():
        text = text.replace(search, replace)
    return text
