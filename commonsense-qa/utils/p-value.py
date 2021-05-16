#Student's t-test
import sys
import json
from numpy.random import seed
from numpy.random import randn
from scipy.stats import ttest_ind, ttest_rel
import numpy as np
import pandas as pd

# seed the random number generator
seed(1)
# generate two independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51

def compare_samples(data1, data2):
    stat, p = ttest_rel(data1, data2)
    print('Statistics={}, p={}'.format(stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)+\n')
    else:
        print('Different distributions (reject H0) +\n')
    return stat, p


def load_pred_gold(test_pred_path):
    preds = []
    labels = []
    dis =[]
    with open(test_pred_path, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            js = json.loads(line.strip(), encoding='utf-8')
            labels.append(js['label'] )
            preds.append(js['pred_label'] )
            if js['pred_label']==js['label']:
                dis.append(1)
            else:
                dis.append(0)

    preds = np.array(preds)
    labels = np.array(labels)
    pred_one_hot = one_hot_label(preds)
    label_one_hot= one_hot_label(labels)
    dis = np.array(dis)

    return dis 
    #return pred_one_hot, label_one_hot

def one_hot_label(a):
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1

    return b

def compute_p_value(cn_prediction, sw_prediction):
    cn_dis = load_pred_gold(cn_prediction)
    sw_dis = load_pred_gold(sw_prediction)

   
    print("compare cn and sw")
    t, p=compare_samples(cn_dis, sw_dis)
    return t, p
   

def csqa_predictions():
    sw_predictions = [
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s19286_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_test.json', 
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s11010_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_test.json', 
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s0_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_test.json', 
    ]
        # './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s21527_g0_pg_full_aatt_pool_swow_entroberta_p1.0_1/predictions_test.json'
        # './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s11010_g0_pg_full_aatt_pool_swow_entroberta_p1.0_1/predictions_test.json', 


    sw_predictions_dev = [
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s19286_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_dev.json', 
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s11010_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_dev.json', 
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s0_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_dev.json', 
    ]
        # './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s11010_g0_pg_full_aatt_pool_swow_entroberta_p1.0_1/predictions_dev.json', 
        # './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s21527_g0_pg_full_aatt_pool_swow_entroberta_p1.0_1/predictions_dev.json'


    cn_predictions=[
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s0_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_0/predictions_test.json',
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s23528_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_3/predictions_test.json',
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s24524_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_2/predictions_test.json',
    ]

    cn_predictions_dev=[
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s0_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_0/predictions_dev.json',
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s23528_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_3/predictions_dev.json', 
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s24524_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_2/predictions_dev.json',
    ]
    return cn_predictions, sw_predictions, cn_predictions_dev, sw_predictions_dev

def obqa_predictions():
    sw_predictions = [
        './saved_models/obqa/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s10806_g0_pg_full_aatt_pool_swow_entroberta_p1.0_3/predictions_test.json',
        './saved_models/obqa/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s17036_g0_pg_full_aatt_pool_swow_entroberta_p1.0_5/predictions_test.json',
        './saved_models/obqa/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s3469_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_test.json',
    ]

    sw_predictions_dev = [
        './saved_models/obqa/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s10806_g0_pg_full_aatt_pool_swow_entroberta_p1.0_3/predictions_dev.json',
        './saved_models/obqa/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s17036_g0_pg_full_aatt_pool_swow_entroberta_p1.0_5/predictions_dev.json',
        './saved_models/obqa/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s3469_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_dev.json',
    ]
    
    cn_predictions = [
       './saved_models/obqa/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s8625_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_1/predictions_test.json',
       './saved_models/obqa/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s14846_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_5/predictions_test.json',
       './saved_models/obqa/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s5863_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_4/predictions_test.json',
    ]

    cn_predictions_dev = [
        './saved_models/obqa/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s8625_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_1/predictions_dev.json',
        './saved_models/obqa/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s14846_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_5/predictions_dev.json',
        './saved_models/obqa/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s5863_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_4/predictions_dev.json',
    ]

    return cn_predictions, sw_predictions, cn_predictions_dev, sw_predictions_dev


if __name__=='__main__':
    dataset=sys.argv[1]
    print(dataset)
    if dataset=='csqa':
        cn_predictions, sw_predictions, cn_predictions_dev, sw_predictions_dev = csqa_predictions()

    elif dataset=='obqa':
        cn_predictions, sw_predictions, cn_predictions_dev, sw_predictions_dev = obqa_predictions()

    results=[]
    for i, x in enumerate(cn_predictions):
        for j, y in enumerate(sw_predictions):
            t, p=compute_p_value(x, y)
            results.append([i, j, t, p])
    
    df = pd.DataFrame(results, columns=["cn_model", "sw_model", "t", "p"])
    print(df)
    #compute_p_value(cn_predictions[0], sw_predictions[0])
    #compute_p_value(cn_predictions[1], sw_predictions[1])
    #compute_p_value(cn_predictions[2], sw_predictions[2])
