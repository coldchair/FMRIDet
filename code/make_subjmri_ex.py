import argparse
import os
import numpy as np
import pandas as pd
from nsd_access import NSDAccess
import scipy.io

from config import NSD_ROOT_DIR, DATA_ROOT_DIR
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )
    return parser.parse_args()


def calc(len):
    return 2 ** len - 1 - len

img_2_fmri_id = [[] for _ in range(73000)]
count = []
add_num = 0

# name : each or ave
def write_index(sharedix, stims, name, save_dir):
    feats = []
    tr_idx = np.zeros(len(stims))
    for idx, s in tqdm(enumerate(stims)): 
        if s in sharedix:
            tr_idx[idx] = 0
        else:
            tr_idx[idx] = 1    
        feats.append(s)
    
    feats = np.stack(feats)

    feats_tr = feats[tr_idx==1]
    feats_te = feats[tr_idx==0]

    global count
    global add_num
    global img_2_fmri_id

    for idx, x in enumerate(feats_tr):
        img_2_fmri_id[x].append(idx)

    print(f'len(feats_tr): {len(feats_tr)}')
    for i in range(73000):
        if (len(img_2_fmri_id[i]) > 1): # mixup
            c = calc(len(img_2_fmri_id[i]))
            count.append(len(img_2_fmri_id[i]))
            add_num += c
            feats_tr = np.concatenate((feats_tr, [i for _ in range(c)]), 0)
    print(f'count: {count}')
    print(f'{add_num} added')
    print(f'len(feats_tr): {len(feats_tr)}')
    

    np.save(f'{save_dir}/index_{name}_tr_ex.npy',feats_tr)
    # np.save(f'{save_dir}/index_{name}_te.npy',feats_te)

def main():
    opt = parse_args()

    subject = opt.subject
    atlasname = 'streams'
    
    nsda = NSDAccess(NSD_ROOT_DIR)
    nsd_expdesign = scipy.io.loadmat(os.path.join(NSD_ROOT_DIR, 'nsddata/experiments/nsd/nsd_expdesign.mat'))

    # Note that most of nsd_expdesign indices are 1-base index!
    # This is why subtracting 1
    sharedix = nsd_expdesign['sharedix'] -1 

    behs = pd.DataFrame()
    for i in range(1,38):
        beh = nsda.read_behavior(subject=subject, 
                                session_index=i)
        behs = pd.concat((behs,beh))

    # Caution: 73KID is 1-based! https://cvnlab.slite.page/p/fRv4lz5V2F/Behavioral-data
    stims_unique = behs['73KID'].unique() - 1
    stims_all = behs['73KID'] - 1

    savedir = os.path.join(DATA_ROOT_DIR, f'mrifeat/{subject}/')

    os.makedirs(savedir, exist_ok=True)

    if not os.path.exists(f'{savedir}/{subject}_stims.npy'):
        np.save(f'{savedir}/{subject}_stims.npy',stims_all)
        np.save(f'{savedir}/{subject}_stims_ave.npy',stims_unique)
    
    write_index(sharedix, stims_all, 'each', savedir)
    # write_index(sharedix, stims_unique, 'ave', savedir)
    # exit(0)

    for i in tqdm(range(1,38), desc = 'reading betas'):
        beta_trial = nsda.read_betas(subject=subject, 
                                session_index=i, 
                                trial_index=[], # empty list as index means get all for this session
                                data_type='betas_fithrf_GLMdenoise_RR',
                                data_format='func1pt8mm')
        if i==1:
            betas_all = beta_trial
        else:
            betas_all = np.concatenate((betas_all,beta_trial),0)    
    
    atlas = nsda.read_atlas_results(subject=subject, atlas=atlasname, data_format='func1pt8mm')
    for roi,val in atlas[1].items():
        print(roi,val)
        if val == 0:
            print('SKIP')
            continue
        else:
            betas_roi = betas_all[:,atlas[0].transpose([2,1,0])==val]

        print(betas_roi.shape)
        
        # Averaging for each stimulus
        betas_roi_ave = []
        for stim in stims_unique:
            stim_mean = np.mean(betas_roi[stims_all == stim,:],axis=0)
            betas_roi_ave.append(stim_mean)
        betas_roi_ave = np.stack(betas_roi_ave)
        print(betas_roi_ave.shape)
        
        # Train/Test Split
        # ALLDATA
        betas_tr = []
        betas_te = []

        for idx,stim in enumerate(stims_all):
            if stim in sharedix:
                betas_te.append(betas_roi[idx,:])
            else:
                betas_tr.append(betas_roi[idx,:])
        
        print(f'train {betas_tr.shape} test {betas_te.shape}')

        betas_tr = np.stack(betas_tr)
        betas_te = np.stack(betas_te)    
        
        # AVERAGED DATA        
        # betas_ave_tr = []
        # betas_ave_te = []
        # for idx,stim in enumerate(stims_unique):
        #     if stim in sharedix:
        #         betas_ave_te.append(betas_roi_ave[idx,:])
        #     else:
        #         betas_ave_tr.append(betas_roi_ave[idx,:])
        # betas_ave_tr = np.stack(betas_ave_tr)
        # betas_ave_te = np.stack(betas_ave_te)    

        scaler = StandardScaler()
        betas_tr = scaler.fit_transform(betas_tr)
        betas_te = scaler.transform(betas_te)

        # scaler = StandardScaler()
        # betas_ave_tr = scaler.fit_transform(betas_ave_tr)
        # betas_ave_te = scaler.transform(betas_ave_te)

        # 给 betas_tr 加 add_num 维
        betas_tr = np.concatenate((betas_tr, np.zeros((add_num, betas_tr.shape[1]))), 0)    
        st = betas_tr.shape[0] - add_num
        
        # do mixup
        for i in range(73000):
            if (len(img_2_fmri_id[i]) > 1):
                l = len(img_2_fmri_id[i])
                assert(l == 2 or l == 3)
                if (l == 2):
                    beta = np.mean(betas_tr[img_2_fmri_id[i]], axis=0)
                    betas_tr[st] = beta
                    st += 1
                elif (l == 3):
                    beta_0 = np.mean(betas_tr[img_2_fmri_id[i]], axis=0)
                    beta_1 = np.mean(betas_tr[img_2_fmri_id[i][0:2]], axis=0)
                    beta_2 = np.mean(betas_tr[img_2_fmri_id[i][1:3]], axis=0)
                    beta_3 = np.mean(betas_tr[[img_2_fmri_id[i][0], img_2_fmri_id[i][2]]], axis=0)
                    # 合并 beta0~3 到 betas_tr
                    betas_tr[st] = beta_0
                    st += 1
                    betas_tr[st] = beta_1
                    st += 1
                    betas_tr[st] = beta_2
                    st += 1
                    betas_tr[st] = beta_3
                    st += 1
        
        print(betas_tr.shape)
                    
        # Save
        np.save(f'{savedir}/{subject}_{roi}_betas_tr_ex.npy',betas_tr)
        # np.save(f'{savedir}/{subject}_{roi}_betas_te.npy',betas_te)
        # np.save(f'{savedir}/{subject}_{roi}_betas_ave_tr.npy',betas_ave_tr)
        # np.save(f'{savedir}/{subject}_{roi}_betas_ave_te.npy',betas_ave_te)


if __name__ == "__main__":
    main()
