'''
@Author: feiyun
@Github: https://github.com/feiyunamy
@Blog: https://blog.feiyunamy.cn
@Date: 2019-10-16 20:05:33
@LastEditors: feiyun
@LastEditTime: 2019-10-21 11:05:37
'''
import logging
import numpy as np
from datetime import datetime
import pprint

def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string

def get_ner_BIO(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
                index_tag = current_label.replace(begin_label,"",1)
            else:
                tag_list.append(whole_tag + ',' + str(i-1))
                whole_tag = current_label.replace(begin_label,"",1)  + '[' + str(i)
                index_tag = current_label.replace(begin_label,"",1)

        elif inside_label in current_label:
            if current_label.replace(inside_label,"",1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '')&(index_tag != ''):
                    tag_list.append(whole_tag +',' + str(i-1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '')&(index_tag != ''):
                tag_list.append(whole_tag +',' + str(i-1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix

def get_ner_BMES(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
            index_tag = current_label.replace(begin_label,"",1)

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(single_label,"",1) +'[' +str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag +',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    # print stand_matrix
    return stand_matrix

def get_correct_num(predict, target):
    start = datetime.now()
    correct_num = 0
    for p in predict:
        p = p.replace('NAME','')
        p_left = eval(p)[0]
        p_right = eval(p)[1]
        for t in target:
            t = t.replace('NAME','')
            t_left = eval(t)[0]
            t_right = eval(t)[1]
            # if p_left == t_left and p_right == t_right:
            if p == t:
                correct_num += 1
                break
            elif abs(p_left - t_left) <= 3 and abs(p_right - t_right) <= 3:
                correct_num += 1
                break
    used_time = (datetime.now() - start).total_seconds()
    return correct_num, used_time

# IOU判断实体相交情况
def get_correct_num_np(target_idxs, predict_idxs, threshold = None):
    # target_idxs = [[[target] * len(predict_idxs)] for target in target_idxs]
    # predict_idxs = [[[predict] * len(target_idxs)] for predict in predict_idxs]
    # p = len(predict_idxs)
    # t = len(target_idxs)
    start = datetime.now()
    # target.shape = t * 1 * 2
    target = np.array(target_idxs)[:, None]
    # predict.shape = 1 * p * 2
    predict = np.array(predict_idxs)[None]
    # sub = predict - target
    # 相交部分长度
    intersec = np.minimum(predict[...,1],target[...,1]) - np.maximum(predict[...,0],target[...,0])
    predict_len = predict[:,:,1] - predict[:,:,0]
    target_len = target[:,:,1] - target[:,:,0]
    all_len = predict_len + target_len - intersec
    iou = intersec / all_len
    if threshold == None:
        # correct_num = (abs(sub[...,0]) + abs(sub[...,1]) == 0).sum()
        correct_num = (iou == 1).sum()
    else:
        correct_num = (iou >= threshold).sum()
        # correct_num = (abs(sub[...,0]) + abs(sub[...,1]) <= threshold).sum()
    used_time = (datetime.now() - start).total_seconds()
    return correct_num, used_time
    
def get_ner_BIO_metrics(predict_labels, target_labels, threshold = None):
    target_entities = get_ner_BIO(target_labels)
    predict_entities = get_ner_BIO(predict_labels)
    correct_num = len(list(set(target_entities).intersection(set(predict_entities))))
    tag_types = set([item[item.index(']') + 1:] for item in target_entities])
    p =  correct_num / len(predict_entities) if len(predict_entities) != 0 else 0.0
    r = correct_num / len(target_entities) if len(target_entities) != 0 else 0.0
    f = (2 * p * r) / (p + r) if p + r != 0 else 0.0
    print(len(target_entities), len(predict_entities), correct_num)    
    all_rs = {'Precision' : p, 'Recall' : r, 'F1': f}
    results = {}
    results['All'] = all_rs
    for tag in tag_types:
        target_idxs = [eval(target[:target.index(']') + 1]) for target in target_entities if tag in target]
        predict_idxs = [eval(predict[:predict.index(']') + 1]) for predict in predict_entities if tag in predict]
        correct_num, _ = get_correct_num_np(target_idxs, predict_idxs)
        predict_num = len(predict_idxs)
        target_num = len(target_idxs)
        precision = correct_num / predict_num if predict_num != 0 else 0.0
        recall = correct_num / target_num if target_num != 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0.0
        rs = {'Precision' : precision, 'Recall' : recall, 'F1': f1}
        results[tag] = rs
    return results
    
if __name__ == '__main__':
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler('info.log')
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    with open('./label_1','r',encoding='utf-8') as f:
        lines = f.readlines()
    target_labels = [line.split()[1] for line in lines if line != '\n']
    predict_labels = [line.split()[2] for line in lines if line != '\n']
    assert len(target_labels) == len(predict_labels), 'ERROR split!'
    target_entities = get_ner_BIO(target_labels)
    predict_entities = get_ner_BIO(predict_labels)

    # 获取不包括tag_name的idx list
    target_idxs = [eval(target[:target.index(']') + 1]) for target in target_entities]
    predict_idxs = [eval(predict[:predict.index(']') + 1]) for predict in predict_entities]

    # 向量化采用IOU计算模糊匹配数目 threshold 不设置值时为精确匹配
    # correct_num, used_time = get_correct_num_np(target_idxs, predict_idxs)
    
    # 结果与标注idx完美匹配的规则
    # correct_entities = list(set(target_entities).intersection(set(predict_entities)))
    # correct_num = len(list(set(target_entities).intersection(set(predict_entities))))
    
    # 结果与标注idx非完美匹配，模糊匹配的非向量化计算方式
    # correct_num, used_time = get_correct_num(predict_entities, target_entities)
    # target_num = len(target_entities)
    # predict_num = len(predict_entities)
    # correct_num = len(correct_entities)
    # num_info = 'predict:{},target:{},correct:{}'.format(predict_num, target_num, correct_num)
    # info = 'Precision:{:.2f}%\tRecall:{:.2f}%\tTIME:{}s'.format(correct_num / predict_num * 100, correct_num / target_num * 100, used_time)
    
    # logger.info(num_info)
    # logger.info(info)

    # print(num_info)
    # print(info)

    rs = get_ner_BIO_metrics(predict_labels, target_labels)
    logger.info(rs)
    print(rs)
    # pprint(rs)
   