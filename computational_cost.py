from modules import gendata, SI_SeqFS_DA, DA_SeqFS
import numpy as np
def main():
    ns_list = [160]
    ns = 20
    nt_list = [80,]
    nt = 10
    truebeta= [0, 0,  0]
    p = len(truebeta)
    print('true beta:', truebeta)

    true_beta_s = np.full((p,1), 2) #source's beta
    true_beta_t = np.array(truebeta).reshape((-1,1)) #target's beta

    K = 2 # select k features
    for ns in ns_list:
        print(f"nt = {nt}")
        for iter in range(6):
        # generate data
            Xs, Xt, Ys, Yt, Sigma_s, Sigma_t = gendata.generate(ns, nt, p, true_beta_s, true_beta_t, fixed_seed=None)

            # apply DA-SeqFS to select features
            list_selected_features = DA_SeqFS.DA_SeqFS(Xs, Ys, Xt, Yt, Sigma_s, Sigma_t,K, method='forward')

            # SI-SeqFS-DA to calculate p-value of the selected features
            # print('Index of selected features:', '{' + ', '.join(map(str, list_selected_features)) + '}')
            # print('Applying SI-SeqFS-DA to calculate p-value of the selected features:')
            j = np.random.randint(0, len(list_selected_features))
            pvalue = SI_SeqFS_DA.SI_SeqFS_DA(Xs, Ys, Xt, Yt, K, Sigma_s, Sigma_t, method='forward', jth = j)
            print(f'    p-value of feature {list_selected_features[j]}: {pvalue}')

if __name__ == "__main__":
    # import time
    # start_time = time.time()
    main()
    # print("Time: ", time.time() - start_time, "seconds")