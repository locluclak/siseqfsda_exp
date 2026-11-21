from modules import gendata, SI_SeqFS_DA, DA_SeqFS

import numpy as np
def main():
    ns_list = [20]
    ns = 50
    nt_list = [10]
    nt = 10
    truebeta= [0, 0, 0, 0, 0]
    p = len(truebeta)
    # print('true beta:', truebeta)
    ps = 6 # the value of beta for source data
    true_beta_s = np.full((p,1), ps) #source's beta
    true_beta_t = np.array(truebeta).reshape((-1,1)) #target's beta

    K = 3 # select k features
    
    for iter in range(120):
    # generate data
        print(f"Iteration {iter+1}:")
        Xs, Xt, Ys, Yt, Sigma_s, Sigma_t = gendata.generate(ns, nt, p, true_beta_s, true_beta_t, fixed_seed=None)

        # apply DA-SeqFS to select features
        list_selected_features = DA_SeqFS.DA_SeqFS(Xs, Ys, Xt, Yt, Sigma_s, Sigma_t, K, method='forward')

        # SI-SeqFS-DA to calculate p-value of the selected features
        # print('Index of selected features:', '{' + ', '.join(map(str, list_selected_features)) + '}')
        # print('Applying SI-SeqFS-DA to calculate p-value of the selected features:')
        j = np.random.randint(0, len(list_selected_features))
        pvalue = SI_SeqFS_DA.SI_SeqFS_DA(Xs, Ys, Xt, Yt, K, Sigma_s, Sigma_t, method='forward', jth = j)
        print(f'    p-value of feature {list_selected_features[j]}: {pvalue}')
        with open(f"new_exp/FPR_pvalues_nt{nt}_ns{ns}_betas{ps}.txt", "a") as f:
            f.write(f"{pvalue}\n")

if __name__ == "__main__":
    # import time
    # start_time = time.time()
    main()
    # print("Time: ", time.time() - start_time, "seconds")