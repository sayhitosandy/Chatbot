import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
people = ('SOTA', 'LSTM', 'GRU', 'SMT')
y_pos = np.arange(len(people))
performance = np.asarray([2.74, 8.04, 10.54, 26.24])
# error = np.random.rand(len(people))

ax.barh(y_pos, performance, height = 0.5, align='center',
        color='orange')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
for i, v in enumerate(performance):
    ax.text(v-2.5, i-0.05, str(v), color='black', fontweight='bold')

ax.set_ylabel('Model -->')
# ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Perplexity -->')
ax.set_title('Comparison of Perplexity values for different Models')

# plt.show()
plt.savefig("Models Comparison.png")