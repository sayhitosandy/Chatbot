import matplotlib.pyplot as plt
import argparse

val_loss_list = []
train_loss_list = []

parser = argparse.ArgumentParser()
parser.add_argument("--path", type = str )

args = parser.parse_args()

p = args.path
vals = p.split('+')

epochs = int(vals[1])-50
batch = int(vals[2])
hidden_layer_dim = int(vals[3])

with open(p) as f:
	for line in f:
		lines=line.split('Epoch')
		ctr=0
		for line in lines:
			if (line != '\n'):
				dash_split=line.split('-')
				print(dash_split)
				val_loss=float(dash_split[-1].split(":")[-1])
				train_loss=float(dash_split[-2].split(":")[-1])
				print(val_loss)
				print(train_loss)
				
				val_loss_list.append(val_loss)
				train_loss_list.append(train_loss)
				
				if ctr==epochs-5:
					break
				ctr+=1

epochs_list = [i for i in range(1, len(train_loss_list)+1)]

plot_name = p.split('.')[0]
# print(plot_name)

trainL, = plt.plot(epochs_list, train_loss_list, 'b-')
valL, = plt.plot(epochs_list, val_loss_list, 'r-')
plt.legend([trainL, (trainL, valL)], ["Training Loss", "Validation Loss"])
# plt.legend([red_dot, (red_dot, white_cross)], ["Attr A", "Attr A+B"])
plt.ylabel('Loss -->')
plt.xlabel('Epochs -->')
# plt.show()

loss_plot_name = "Loss_" + plot_name + ".png"
# print(loss_plot_name)
plt.savefig(loss_plot_name)
plt.gcf().clear()

train_perp_list = [2**i for i in train_loss_list]
val_perp_list = [2**i for i in val_loss_list]

trainPL, = plt.plot(epochs_list, train_perp_list, 'b-')
valPL, = plt.plot(epochs_list, val_perp_list, 'r-')
plt.legend([trainPL, (trainPL, valPL)], ["Training Perplexity", "Validation Perplexity"])
# plt.legend([red_dot, (red_dot, white_cross)], ["Attr A", "Attr A+B"])
plt.ylabel('Perplexity -->')
plt.xlabel('Epochs -->')
# plt.show()

perp_plot_name = "Perp_" + plot_name + ".png"
plt.savefig(perp_plot_name)
plt.gcf().clear()