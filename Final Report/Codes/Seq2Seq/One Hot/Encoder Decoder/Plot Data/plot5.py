import matplotlib.pyplot as plt

f = [
 open("LSTM+30+4+512+NoReversal.txt", 'r'),
 # open("LSTM+75+16+2+NoReversal.txt", 'r'),
 # open("LSTM+150+32+2+NoReversal.txt", 'r'),
 open("LSTM+200+64+512+NoReversal.txt", 'r')
]

valL = []
epochs_list = [i for i in range(1, 200)]

for i in f:
	val_loss_list = []
	sp = i.name.split('+')
	epochs = int(sp[1])
	batch = int(sp[2])
	hidden_layer_dim = 2

	print(i.name)

	for line in i:
		# print(line)
		lines = line.split("Epoch")
		
		ctr = 0

		for _line in lines:
			if (_line != '\n'):
				dash_split = _line.split('-')
				# print(dash_split)

				val_loss = float(dash_split[-1].split(':')[-1])
				val_loss_list.append(2**(val_loss))

				if (ctr == epochs-5):
					break
				ctr += 1

		valL.append(val_loss_list)

plot_name = "Batch Size Study (Latent Dimensions 512).png"

colorLabel = {	0:'black',
				1:'red',
				2:'gold',
				3:'blue',
				4:'g',
				5:'violet',
				6:'crimson',
				7:'grey',
				8:'pink',
				9:'navy'}

for i in range(len(valL)):
	# print(len(epochs_list[:len(valL[i])])) 	
	plt.plot(epochs_list[:len(valL[i])], valL[i], colorLabel[i])

axes = plt.gca()
# axes.set_ylim([4,20])
axes.set_xlim([0,40])

plt.legend(["4", "64"], loc = 'upper right', fontsize = 10)

plt.ylabel('Perplexity -->')
plt.xlabel('Epochs -->')
plt.title("Comparitive Study of Batch Sizes in LSTM with Latent Dimensions = 512")

# plt.show()
plt.savefig(plot_name)
