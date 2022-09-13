import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

COLORS = [((0, 40, 150), (77, 180, 255)), ((204, 0, 0), (255, 150, 150)), ((25, 102, 25), (111, 220, 111)), ((179, 179, 0), (255, 255, 102))] # DARk, LIGHT
COLORS = np.asarray(COLORS) / 255.0

def moving_average(x, N = 200):
    return np.convolve(x, np.ones((N,))/N, mode='valid')

def moving_stddev(x, N = 200):
    x = np.asarray(x)
    avg = np.convolve(x, np.ones((N,))/N, mode = 'valid')
    var = np.sqrt(np.convolve(x**2, np.ones((N,))/N, mode = 'valid') - np.square(avg))
    return var + avg, avg - var

def process_dict(log_dict, output_file_path):
    MAXPLOT = 11500
    fills = {'lb': [], 'ub': [], 'color': []}
    hard_plots = {'x': [], 'color': [], 'label': []}
    for i, key in enumerate(log_dict):
        x = moving_average(log_dict[key])
        hard_plots['x'].append(x)
        hard_plots['label'].append(key)
        hard_plots['color'].append(COLORS[i][0])
        # plt.plot(x, label = key, color = COLORS[i][0], linewidth = 2)
        ub, lb = moving_stddev(log_dict[key])
        fills['ub'].append(ub)
        fills['lb'].append(lb)
        fills['color'].append(COLORS[i][0])
        # plt.fill_between(list(range(len(x))), y1 = ub, y2 = lb, color = COLORS[i][1])
    for ub, lb, color in zip(fills['lb'], fills['ub'], fills['color']):
        plt.fill_between(list(range(len(ub[:MAXPLOT]))), y1 = ub[:MAXPLOT], y2 = lb[:MAXPLOT], color = color[:MAXPLOT], alpha = 0.15)

    for x, color, label in zip(hard_plots['x'], hard_plots['color'], hard_plots['label']):
        plt.plot(x[:MAXPLOT], label = label, color = color[:MAXPLOT], linewidth = 2)


    plt.legend(loc = 'upper left')
    # plt.show()
    plt.savefig(output_file_path)

if __name__ == '__main__':
    from argparse import ArgumentParser
    argparse = ArgumentParser()
    argparse.add_argument('--output_file_path', type = str, required = True)
    argparse.add_argument('-l','--log_files', nargs='+', help='<Required> Set flag', required = True)
    argparse.add_argument('-r','--plot_names', nargs='+', help='<Required> Set flag', required = True)
    args = argparse.parse_args()

    assert len(args.plot_names) == len(args.log_files), "There should be an identiacal number of plot names and log files"

    log_dict = {}
    from tensorflow.core.util import event_pb2
    for log_file, plot_name in zip(args.log_files, args.plot_names):
        serialized_examples = tf.data.TFRecordDataset(log_file)
        print(plot_name)
        for serialized_example in serialized_examples:
            event = event_pb2.Event.FromString(serialized_example.numpy())
            for value in event.summary.value:
                if plot_name not in log_dict:
                    log_dict[plot_name] = []
                t = tf.make_ndarray(value.tensor)
                log_dict[plot_name].append(t)

    process_dict(log_dict, args.output_file_path)
