import matplotlib.pyplot as plt
# Helper: Plot de algumas amostras do sinal
def plot_sig_matrix(title, signal, ncols=3, nrows=2):
    fig, axs = plt.subplots(nrows, ncols, figsize=(20,4))
    for i in range(nrows):
        for j in range(ncols):
            axs[i,j].plot(signal[i*3+j])
    plt.suptitle(title)

def plot_distributions_per_class(title, features, class_filter):
    fig, axs = plt.subplots(1, len(features), figsize=(20,3))
    for i, f_data in enumerate(features):
        labels = []
        plot_data = []
        for ck in class_filter:
            labels.append(ck)
            plot_data.append(f_data[class_filter[ck]][:].flatten())
        axs[i].boxplot(plot_data, labels=labels , showfliers=False)
        axs[i].set_title(f'Sensor {i}')
    plt.suptitle(title)
    
def plot_freq_samples(title, freq_data, class_filter, samples=50):
    fig, axs = plt.subplots(len(class_filter), len(freq_data), figsize=(30,30))        
    for i, ck in enumerate(class_filter):
        for j, f_data in enumerate(freq_data):
            axs[i, j].imshow(freq_data[j][class_filter[ck]][:samples], vmin=0, vmax=1, cmap='hot')
            axs[i, j].set_title(f's{j}, {ck}')
    plt.suptitle(title)
    plt.tight_layout()