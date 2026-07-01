with open('stage18e_d3_pilot_train.py', 'r') as f:
    content = f.read()

old_data_load = '''    data = np.load("dataset_3d_pilot_masks.npz")
    g_exact_full = data['g_exact']
    jam_bodies_full = data['jam_bodies']
    headings_full = data['headings']
    
    N_POINTS = len(headings_full)'''

new_data_load = '''    data = np.load("dataset_shadow_100k_polar_32el.npz")
    inputs = data['inputs']
    labels_polar = data['labels']
    
    mag = labels_polar[:, 0::2]
    phase = labels_polar[:, 1::2]
    g_exact_full = mag * np.exp(1j * phase)
    jam_bodies_full = inputs
    
    headings_full = np.rad2deg(np.arctan2(inputs[:, 1], inputs[:, 0]))
    headings_full = np.where(headings_full < 0, headings_full + 360.0, headings_full)
    
    N_POINTS = len(headings_full)'''

content = content.replace(old_data_load, new_data_load)

old_loop = '''    for i, h in enumerate(headings_full):
        valid_indices.append(i)
        margin_weights_list.append(1.0)'''

new_loop = '''    for i, h in enumerate(headings_full):
        if h <= 10.0 or h >= 350.0:
            continue
        valid_indices.append(i)
        
        if h <= 30.0 or h >= 330.0:
            margin_weights_list.append(0.1)
        else:
            margin_weights_list.append(1.0)'''

content = content.replace(old_loop, new_loop)

content = content.replace('EPOCHS = 300', 'EPOCHS = 75')
content = content.replace('BATCH_SIZE = 64', 'BATCH_SIZE = 1024')
content = content.replace('siren_beamformer_d3_cov_K5_3D_no_excl.pt', 'siren_beamformer_d3_cov_K5_100k_physics.pt')

with open('stage18s_train_100k_physics.py', 'w') as f:
    f.write(content)
