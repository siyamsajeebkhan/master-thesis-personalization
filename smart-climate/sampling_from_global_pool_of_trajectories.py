    # # Define a custom sorting key function
    # def sorting_key(dictionary):
    #     return (dictionary['traj_no'], dictionary['timestamp'])
    
    # trajectories_list = []
    # for traj_no, traj in enumerate(trajectories):
    #     trajectories_list += [{f'traj_no': traj_no, 'timestamp': t, 'state': obs[0], 'action': obs[1], 'reward': obs[2], 'terminal': obs[3]} for t, obs in enumerate(zip(traj['observations'], traj['actions'], traj['rewards'], traj['terminals']))]
    # trajectories_arr = np.array(trajectories_list)    
    # num_samples = trajectories_arr.shape[0]
    
    # train_idx = np.random.choice(num_samples, size=int(num_samples*0.8), replace=False)
    # test_idx = np.setdiff1d(np.arange(num_samples), train_idx)

    
    # trajs_train = sorted(trajectories_arr[train_idx], key=sorting_key)
    # trajs_test = sorted(trajectories_arr[test_idx], key=sorting_key)
    
    # print(f"Preparing the train trajectories")
    # keys = []
    # for traj in trajs_train:
    #     keys.append(traj['traj_no'])
    # keys = set(keys)
    # trajs_train_df = pd.DataFrame(trajs_train)

    # trajectories = []
    # for key in tqdm(keys):
    #     df_key = trajs_train_df[trajs_train_df['traj_no']==key]
    #     obss = np.array(df_key['state'].tolist())
    #     actions = np.array(df_key['action'].tolist())
    #     rewards = np.array(df_key['reward'].tolist())
    #     terminals = np.array(df_key['terminal'].tolist())
    #     trajectories.append(
    #         {
    #             'observations': obss,
    #             'actions': actions,
    #             'rewards': rewards,
    #             'terminals': terminals,
    #         }
    #     )
    # # Prepare the test set
    # print(f"Preparing the test trajectories")
    # keys = []
    # for traj in trajs_test:
    #     keys.append(traj['traj_no'])
    # keys = set(keys)
    # trajs_test_df = pd.DataFrame(trajs_test)
    
    # test_trajectories = []
    # for key in tqdm(keys):
    #     df_key = trajs_test_df[trajs_test_df['traj_no']==key]
    #     obss = np.array(df_key['state'].tolist())
    #     actions = np.array(df_key['action'].tolist())
    #     rewards = np.array(df_key['reward'].tolist())
    #     terminals = np.array(df_key['terminal'].tolist())
    #     terminals[-1] = True
    #     test_trajectories.append(
    #         {
    #             'observations': obss,
    #             'actions': actions,
    #             'rewards': rewards,
    #             'terminals': terminals,
    #         }
    #     )
    # print(f"min events per drive in train set: {trajs_train_df['traj_no'].value_counts().min()} and in the test set: {trajs_test_df['traj_no'].value_counts().min()}")