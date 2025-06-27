import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import networkx as nx
#import community as community_louvain
import os

## AVG DATA
# read npz files
all_central_chans = np.load("all_central_chans.npz")
TimeFreq_English = np.load("TimeFreq_english.npz")
TimeFreq_Spanish = np.load("TimeFreq_spanish.npz")

# read csv files
channel_info = np.genfromtxt("channel_info.csv", delimiter=",", dtype=None)

# get the data
central_chans = all_central_chans['channels']
esp_mor_avg = TimeFreq_Spanish['esp_mor'][:,central_chans,:]# gravacions, canals, temps
esp_syl_avg = TimeFreq_Spanish['esp_syl'][:,central_chans,:]
esp_str_avg = TimeFreq_Spanish['esp_str'][:,central_chans,:]
eng_mor_avg = TimeFreq_English['eng_mor'][:,central_chans,:]
eng_syl_avg = TimeFreq_English['eng_syl'][:,central_chans,:]
eng_str_avg = TimeFreq_English['eng_str'][:,central_chans,:]

channel_info = channel_info[central_chans]

# mix esp and eng
mor_avg = np.concatenate((eng_mor_avg, esp_mor_avg), axis=0)
syl_avg = np.concatenate((eng_syl_avg, esp_syl_avg), axis=0)
str_avg = np.concatenate((eng_str_avg, esp_str_avg), axis=0)

# mix all
all_avg = np.concatenate((mor_avg, syl_avg, str_avg), axis=0)

# Create a dictionary where the key is a number from 1 to 37 and the value is the channel name
channel_dict = {
    (channel_info[idx][1].decode('utf-8') if isinstance(channel_info[idx][1], bytes)
     else channel_info[idx][1]): idx for idx in range(37)
}

coordenades_dict = {}
for i in range(37):
    coordenades_dict[i] = (float(channel_info[i][2]), float(channel_info[i][3]))

       
# invert the dictionary
channel_dict2 = {v: k for k, v in channel_dict.items()}

canals_frontals = ['AF3', 'F1', 'F3', 'FC5', 'FC3', 'FC1', 'AF4', 'F2', 'F4', 'FC6', 'FC4', 'FC2', 'Fz']
canals_esquerra = ['CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'CPz', 'Pz', 'C1', 'C3', 'C5', 'Cz']
canals_dreta = ['CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'C2', 'C4', 'C6']

## TRIAL DATA

# Iterate over files in the 'trials' folder and select those ending with '.npz'
npz_files = [f for f in os.listdir('trials') if f.endswith('.npz')]
data = []
for fname in npz_files:
    filepath = os.path.join('trials', fname)
    file = np.load(filepath)
    data.append(file)

# Introduïm totes les dades en diferents llistes, que contindran elements amb les dimensions següents:
# [trials, channels, frequencies, time]

eng_mor = []
eng_syl = []
eng_str = []
esp_mor = []
esp_syl = []
esp_str = []

for e in data:
    if 'eng_mor' in e:
        eng_mor.append(e['eng_mor'][:,central_chans,16:29,30:])
        eng_syl.append(e['eng_syl'][:,central_chans,16:29,30:])
        eng_str.append(e['eng_str'][:,central_chans,16:29,30:])
    if 'esp_mor' in e:
        esp_mor.append(e['esp_mor'][:,central_chans,16:29,30:])
        esp_syl.append(e['esp_syl'][:,central_chans,16:29,30:])
        esp_str.append(e['esp_str'][:,central_chans,16:29,30:])

# AVG de les freqüències
for type in [eng_mor, eng_syl, eng_str, esp_mor, esp_syl, esp_str]:
    for i in range(len(type)):
        type[i] = np.nanmean(type[i], axis=2)

all = np.concatenate((np.concatenate(eng_mor), np.concatenate(eng_syl), np.concatenate(eng_str),np.concatenate(esp_mor),np.concatenate(esp_syl),np.concatenate(esp_str)), axis=0)

#convert a list of arrays to a 3D array
eng_mor = np.concatenate(eng_mor, axis=0)
eng_syl = np.concatenate(eng_syl, axis=0)
eng_str = np.concatenate(eng_str, axis=0)
esp_mor = np.concatenate(esp_mor, axis=0)
esp_syl = np.concatenate(esp_syl, axis=0)
esp_str = np.concatenate(esp_str, axis=0)

both_mor = np.concatenate((eng_mor, esp_mor), axis=0)
both_syl = np.concatenate((eng_syl, esp_syl), axis=0)
both_str = np.concatenate((eng_str, esp_str), axis=0)



## FUNCIONS DE VISUALITZACIO

def plot_subject(set, subject_idx, channel_info=channel_info):
    time_ms_label = "time (ms)"
    channel_label = "channel"
    channel_tick_fontsize = 8

    # Define time range and resolution
    time_start = -1000  # -1 second in ms
    time_end = 3500     # 3.5 seconds in ms
    time_resolution = 500  # 50 ms
    time_ticks = np.arange(time_start, time_end + time_resolution, time_resolution)
    time_tick_positions = np.linspace(0, set.shape[2] - 1, len(time_ticks))

    plt.imshow(set[subject_idx, :, :], aspect='auto')
    plt.xlabel(time_ms_label)
    plt.ylabel(channel_label)
    plt.xticks(
        ticks=time_tick_positions,
        labels=[f"{int(tick)}" for tick in time_ticks],
        fontsize=channel_tick_fontsize
    )
    plt.yticks(
        ticks=np.arange(37),
        labels=[
            channel_info[idx][1].decode('utf-8') if isinstance(channel_info[idx][1], bytes)
            else channel_info[idx][1] for idx in range(37)
        ],
        fontsize=channel_tick_fontsize
    )
    plt.title(f"Subject {subject_idx}")
    plt.colorbar()
    plt.show()


def plot_correlation_matrix(set, subject_idx):
    plt.imshow(np.corrcoef(set[subject_idx, :, :]), aspect='auto')
    plt.colorbar()
    plt.title("Matriz de Correlación")
    plt.show()

def plot_channel(set, channel):
    plt.imshow(set[:, channel, :], aspect='auto')
    plt.xlabel("time")
    plt.ylabel("subject id")
    plt.colorbar()
    plt.show()



def plot_channels(set, subject = 'all', channel_list = ['F3', 'P5', 'P6'], data_label = None):
    channels = list([channel_dict[channel] for channel in channel_list])

    # Define time range and resolution
    time_start = -1000  # -1 second in ms
    time_end = 3500     # 3.5 seconds in ms
    time_resolution = 500  # 50 ms
    time_ticks = np.arange(time_start, time_end + time_resolution, time_resolution)
    time_tick_positions = np.linspace(0, set.shape[2] - 1, len(time_ticks))

    plt.figure()
    if subject == 'all':
        set = np.sum(set, axis=0)
    else:
        set = set[subject, :, :]

    for ch in channels:
        plt.plot(set[ch, :], label=f"Channel {list(channel_dict.keys())[list(channel_dict.values()).index(ch)]}")
    plt.xlabel("time (ms)")
    plt.ylabel("power")
    plt.xticks(
        ticks=time_tick_positions,
        labels=[f"{int(tick)}" for tick in time_ticks]
    )
    plt.title(f"Subject: {subject} - Set: {data_label}")
    plt.suptitle(f"Channels: {channel_list} ")
    plt.legend()
    plt.show()



def plot_global(data_set, channels=(channel_dict.keys()), data_label = None):
    channels = list([channel_dict[channel] for channel in channels])
    time_ms_label = "time (ms)"
    channel_label = "channel"
    channel_tick_fontsize = 8

    # Define time range and resolution
    time_start = -1000  # -1 second in ms
    time_end = 3500     # 3.5 seconds in ms
    time_resolution = 500  # 50 ms
    time_ticks = np.arange(time_start, time_end + time_resolution, time_resolution)
    time_tick_positions = np.linspace(0, data_set.shape[2] - 1, len(time_ticks))
    
    data_set = data_set[:, channels, :]
    summed_data = np.sum(data_set, axis=0)  # Sum across all subjects
    plt.imshow(summed_data, aspect='auto')
    plt.xlabel(time_ms_label)
    plt.ylabel(channel_label)
    plt.xticks(
        ticks=time_tick_positions,
        labels=[f"{int(tick)}" for tick in time_ticks],
        fontsize=channel_tick_fontsize
    )
    plt.yticks(
        ticks=np.arange(len(channels)),
        labels=[list(channel_dict.keys())[list(channel_dict.values()).index(idx)] for idx in channels],
        fontsize=channel_tick_fontsize
    )
    plt.title(f"Sum of All Subjects - Set: {data_label}")
    plt.colorbar()
    plt.show()

# Imprimir un scatter plot amb 
def topoplot(pintats_list, channel_info=channel_info):

    x = [channel_info[i][2].astype(float) for i in range(len(channel_info))]
    y = [channel_info[i][3].astype(float) for i in range(len(channel_info))]
    names = [channel_info[i][1] for i in range(len(channel_info))]

    # Define a list of colors for different groups
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta']

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, color="skyblue", s=400, label="Unselected Channels")

    for idx, pintats in enumerate(pintats_list):
        index_pintats = [channel_dict[channel] for channel in pintats]
        xpintats = [x[i] for i in index_pintats]
        ypintats = [y[i] for i in index_pintats]
        color = colors[idx % len(colors)]  # Cycle through colors if more groups than colors
        plt.scatter(xpintats, ypintats, color=color, s=400, label=f"Group {idx + 1}")

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Topoplot')
    # Add labels to the points
    for i, txt in enumerate(names):
        plt.annotate(txt, (x[i], y[i]), fontsize=8, ha='center', va='center')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend out of the box
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()


def dispersio(set):
    data = set.flatten(set, axis =0)
    data  = data.flatten()

def plot_mean_sem_std(data1, data2, data3, deverror = 'error' , engesp = 'both'):
    # Simulem les teves dades (substitueix per les teves pròpies matrius)
    data_g1 = data1.reshape(data1.shape[0]*data1.shape[1],data1.shape[2])

    data_g2 = data2.reshape(data2.shape[0]*data2.shape[1],data2.shape[2])

    data_g3 = data3.reshape(data3.shape[0]*data3.shape[1],data3.shape[2])

    # Crear eix temporal (ex. 60 punts entre -500 i 2500 ms)
    time = np.linspace(500, 3000, 60)

    # Calcular mitjana i error estàndard
    mean_mora = data_g1.mean(axis=0)
    mean_syll = data_g2.mean(axis=0)
    mean_stress = data_g3.mean(axis=0)

    if deverror == 'error':
        sem_g1 = data_g1.std(axis=0) / np.sqrt(data_g1.shape[0])
        sem_g2 = data_g2.std(axis=0) / np.sqrt(data_g2.shape[0])
        sem_g3 = data_g3.std(axis=0) / np.sqrt(data_g3.shape[0])
    elif deverror == 'std':
        sem_g1 = data_g1.std(axis=0)
        sem_g2 = data_g2.std(axis=0)
        sem_g3 = data_g3.std(axis=0)

    # Dibuixar el gràfic
    plt.figure(figsize=(10, 6))

    # Àrees d’error
    if deverror != 'none':
        plt.fill_between(time, mean_mora - sem_g1, mean_mora + sem_g1, color='blue', alpha=0.2)
        plt.fill_between(time, mean_syll - sem_g2, mean_syll + sem_g2, color='green', alpha=0.2)
        plt.fill_between(time, mean_stress - sem_g3, mean_stress + sem_g3, color='orange', alpha=0.2)

    # Línies de mitjana
    if engesp in ['both', 'eng', 'esp']:
        plt.plot(time, mean_mora, color='blue', label='Mora')
        plt.plot(time, mean_syll, color='green', label='Syllable')
        plt.plot(time, mean_stress, color='orange', label='Stress')
    elif engesp == 'canals': 
        plt.plot(time, mean_mora, color='blue', label='Canals frontals')
        plt.plot(time, mean_syll, color='green', label='Canals esquerra')
        plt.plot(time, mean_stress, color='orange', label='Canals Dreta')


    # Estètica
    plt.xlabel('Time in milliseconds')
    plt.ylabel('Average Power')
    if deverror == 'error': 
        if engesp == 'eng':
            plt.title('Average Power over time with SEM (English)')
        elif engesp == 'esp':
            plt.title('Average Power over time with SEM (Spanish)')
        else:
            plt.title('Average Power over time with SEM')
    else:
        plt.title('Average Power over time with STD')
    
    plt.legend()
    plt.grid(True)

    plt.xticks(np.linspace(500, 3000, 6))  # Set x-ticks from 500 to 3000 with 6 intervals

    if deverror == 'error': 
        plt.ylim(min(mean_stress.min(), mean_mora.min()) - sem_g1.max() - 0.5, max(mean_mora.max(), mean_stress.max()) + sem_g1.max() + 0.5)
    else:
        plt.ylim(min(mean_stress.min(), mean_mora.min(), mean_syll.min()) - 0.5, max(mean_mora.max(), mean_stress.max(), mean_syll.max()) + 0.5)
    plt.tight_layout()
    plt.show()