"""LSTM vs GRU"""
import matplotlib.pyplot as plt
import argparse

val_loss_list = []
val_loss_list2 = []

parser = argparse.ArgumentParser()
parser.add_argument("--path", type = str)
parser.add_argument("--path2", type = str)

args = parser.parse_args()

p = args.path
vals = p.split('+')

q = args.path2

epochs = int(vals[1])
batch = int(vals[2])
hidden_layer_dim = int(vals[3])

with open(p) as f:
	for line in f:
		lines=line.split('Epoch')
		ctr=0
		for line in lines:
			if (line != '\n' and line != ''):
				dash_split=line.split('-')
				print(dash_split)
				val_loss=float(dash_split[-1].split(":")[-1])
				val_loss_list.append(val_loss)
				
				if ctr==epochs-5:
					break
				ctr+=1

with open(q) as f:
	for line in f:
		lines=line.split('Epoch')
		ctr=0
		for line in lines:
			if (line != '\n'):
				dash_split=line.split('-')
				# print(dash_split)
				val_loss=float(dash_split[-1].split(":")[-1])
				val_loss_list2.append(val_loss)
				
				if ctr==epochs-5:
					break
				ctr+=1

epochs_list = [i for i in range(1, len(val_loss_list)+1)]

plot_name = "Comparitive Study of LSTM and GRU"

LSTM, = plt.plot(epochs_list, val_loss_list, 'r-')
GRU, = plt.plot(epochs_list, val_loss_list2[:len(val_loss_list)], 'g-')
plt.legend([LSTM, (LSTM, GRU)], ["LSTM", "GRU"])

axes = plt.gca()
# axes.set_ylim([4,20])
axes.set_xlim([0,800])

plt.ylabel('Validation Loss -->')
plt.xlabel('Epochs -->')
# plt.show()

loss_plot_name = plot_name + ".png"
plt.savefig(loss_plot_name)