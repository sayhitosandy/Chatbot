import matplotlib.pyplot as plt

f = [
 open("LSTM+200+64+4+NoReversal.txt", 'rb'),
 open("LSTM+200+64+8+NoReversal.txt", 'rb'),
 open("LSTM+200+64+16+NoReversal.txt", 'rb'),
 open("LSTM+200+64+32+NoReversal.txt", 'rb'),
 open("LSTM+200+64+64+NoReversal.txt", 'rb'),
 open("LSTM+200+64+128+NoReversal.txt", 'rb'),
 open("LSTM+200+64+256+NoReversal.txt", 'rb'),
 open("LSTM+200+64+512+NoReversal.txt", 'rb')
]

valL = []

for i in f:
	val_loss_list = []
	epochs = 200
	batch = 64
	hidden_layer_dim = int(i.name.split('+')[3])

	for line in i:
		# print(line)
		print(i.name)
		lines = line.split("Epoch".encode('ascii'))
		ctr = 0

		for _line in lines:
			if (_line != '\n'):
				dash_split = _line.split('-'.encode('ascii'))
				print(dash_split)

				val_loss = float(dash_split[-1].split(':'.encode('ascii'))[-1])
				# print(val_loss)

				val_loss_list.append(2**(val_loss))

				if (ctr == epochs-5):
					break
				ctr += 1

	valL.append(val_loss_list)

epochs_list = [i for i in range(1, len(valL[0]) + 1)]
plot_name = "LatentDimensionStudy(Batch64).png"

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
	plt.plot(epochs_list, valL[i], colorLabel[i])

axes = plt.gca()
axes.set_ylim([4,20])
axes.set_xlim([0,200])

plt.legend(["512", "256", "128", "64", "32", "16", "8", "4"], loc = 'upper right', fontsize = 10)

# plt.legend([lab[0], (lab[i] for i in range(len(valL)))], ["4", "8", "16", "32", "64", "128", "256", "512"])
plt.ylabel('Perplexity -->')
plt.xlabel('Epochs -->')
plt.title("Comparitive Study of Latent Dimensions in LSTM with Batch Size = 64")

# plt.show()
plt.savefig(plot_name)
