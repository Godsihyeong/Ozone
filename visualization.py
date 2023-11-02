
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager, rc

ht_data = df[[df.columns[i] for i in [0,1,3,4,5,6]]]

font_path = "C:\Windows\Fonts\malgun.ttf"
font = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font)

colormap = plt.cm.PuBu
plt.figure(figsize=(8,8))
plt.title('변수 간 상관계수', y = 1.05, size = 15)
plt.tight_layout()
sns.heatmap(round(ht_data.astype(float).corr(),2), linewidths=0.1, vmax=1,square=True,cmap=colormap,linecolor='white',annot=True, annot_kws={'size':16})
fig, axes = plt.subplots(2,2, figsize = (8,8))


plt.suptitle('오후 오존 농도와 변수간의 관계', fontsize = 15)

ax1 = axes[0,0]
ax1.scatter(df['오전 오존 농도'], df['오후 오존 농도'], s = 2, color = 'g')
ax1.set_title('오전 오존 농도')
z = np.polyfit(df['오전 오존 농도'],df['오후 오존 농도'],deg=1)
p = np.poly1d(z)
ax1.plot(df['오전 오존 농도'], p(df['오전 오존 농도']), 'r-')
ax1.text(0.04, 0.02, f'y = {round(z[0],2)}x+{round(z[1],2)}', color = 'black', fontsize=15)


ax2 = axes[0,1]
ax2.scatter(df['오전 이산화질소 농도 평균'], df['오후 오존 농도'], s = 2, color = 'g')
ax2.set_title('오전 이산화질소 농도')
z = np.polyfit(df['오전 이산화질소 농도 평균'],df['오후 오존 농도'],deg=1)
p = np.poly1d(z)
ax2.plot(df['오전 이산화질소 농도 평균'], p(df['오전 이산화질소 농도 평균']), 'r-')
ax2.text(0.035, 0.02, f'y = {round(z[0],2)}x+{round(z[1],2)}', color = 'black', fontsize=15)

ax3 = axes[1,0]
ax3.scatter(df['오전 기온 평균'], df['오후 오존 농도'], s = 2, color = 'g')
ax3.set_title('오전 기온 평균')
z = np.polyfit(df['오전 기온 평균'],df['오후 오존 농도'],deg=1)
p = np.poly1d(z)
ax3.plot(df['오전 기온 평균'], p(df['오전 기온 평균']), 'r-')
ax3.text(10, 0.1, f'y = {round(z[0],3)}x+{round(z[1],2)}', color = 'black', fontsize=15)


ax4 = axes[1,1]
ax4.scatter(df['오전 습도 평균'], df['오후 오존 농도'], s = 2, color = 'g')
ax4.set_title('오전 습도 평균')
z = np.polyfit(df['오전 습도 평균'],df['오후 오존 농도'],deg=1)
p = np.poly1d(z)
ax4.plot(df['오전 습도 평균'], p(df['오전 습도 평균']), 'r-')
ax4.text(20, 0.02, f'y = {round(z[0],3)}x+{round(z[1],2)}', color = 'black', fontsize=15)

plt.tight_layout()


import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize = (8,8))

ax1 = axes[0,0]
ax1.boxplot(df['오전 오존 농도'])
ax1.set_title('오전 오전 농도')

ax2 = axes[0,1]
ax2.boxplot(df['오전 이산화질소 농도 평균'])
ax2.set_title('오전 이산화질소 농도 평균')

ax3 = axes[1,0]
ax3.boxplot(df['오전 기온 평균'])
ax3.set_title('오전 기온 평균')

ax4 = axes[1,1]
ax4.boxplot(df['오전 습도 평균'])
ax4.set_title('오전 습도 평균')

plt.suptitle('변수들의 Boxplot', fontsize = 15)
plt.savefig('argBoxplot')
fig, axes = plt.subplots(2,2, figsize = (8,8))



plt.suptitle('오후 오존 농도와 변수간의 관계', fontsize = 15)

ax1 = axes[0,0]
ax1.scatter(df['오전 오존 농도'], df['오후 오존 농도'], s = 2, color = 'g')
ax1.set_title('오전 오존 농도')
z = np.polyfit(df['오전 오존 농도'],df['오후 오존 농도'],deg=1)
p = np.poly1d(z)
ax1.plot(df['오전 오존 농도'], p(df['오전 오존 농도']), 'r-')
ax1.text(0.04, 0.02, f'y = {round(z[0],2)}x+{round(z[1],2)}', color = 'black', fontsize=15)


ax2 = axes[0,1]
ax2.scatter(df['오전 이산화질소 농도 평균'], df['오후 오존 농도'], s = 2, color = 'g')
ax2.set_title('오전 이산화질소 농도')
z = np.polyfit(df['오전 이산화질소 농도 평균'],df['오후 오존 농도'],deg=1)
p = np.poly1d(z)
ax2.plot(df['오전 이산화질소 농도 평균'], p(df['오전 이산화질소 농도 평균']), 'r-')
ax2.text(0.035, 0.02, f'y = {round(z[0],2)}x+{round(z[1],2)}', color = 'black', fontsize=15)

ax3 = axes[1,0]
ax3.scatter(np.log(df['오전 기온 평균']), df['오후 오존 농도'], s = 2, color = 'g')
ax3.set_title('오전 기온 평균')
z = np.polyfit(np.log(df['오전 기온 평균']),df['오후 오존 농도'],deg=1)
p = np.poly1d(z)
ax3.plot(np.log(df['오전 기온 평균']), p(np.log(df['오전 기온 평균'])), 'r-')
ax3.set_xlim(2,3.5)
ax3.text(2.22, 0.1, f'y = {round(z[0],3)}x+{round(z[1],2)}', color = 'black', fontsize=15)


ax4 = axes[1,1]
ax4.scatter(np.log(df['오전 습도 평균']), df['오후 오존 농도'], s = 2, color = 'g')
ax4.set_title('오전 습도 평균')
z = np.polyfit(np.log(df['오전 습도 평균']),df['오후 오존 농도'],deg=1)
p = np.poly1d(z)
ax4.plot(np.log(df['오전 습도 평균']), p(np.log(df['오전 습도 평균'])), 'r-')
ax4.set_xlim(3,4.67)
ax4.text(3.22, 0.02, f'y = {round(z[0],3)}x+{round(z[1],2)}', color = 'black', fontsize=15)

plt.tight_layout()
plt.savefig('산점도와 회귀선')


fig, axes = plt.subplots(2,2, figsize = (8,8))

plt.suptitle('변수들의 Histogram', fontsize = 15)

ax1 = axes[0,0]
ax1.hist(df['오전 오존 농도'], color = 'y', edgecolor = 'k')
ax1.set_title('오전 오존 농도')

ax2 = axes[0,1]
ax2.hist(df['오전 이산화질소 농도 평균'], color = 'y', edgecolor = 'k')
ax2.set_title('오전 이산화질소 농도')

ax3 = axes[1,0]
ax3.hist(df['오전 기온 평균'], color = 'y', edgecolor = 'k')
ax3.set_title('오전 기온 평균')

ax4 = axes[1,1]
ax4.hist(df['오전 습도 평균'], color = 'y', edgecolor = 'k')
ax4.set_title('오전 습도 평균')

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2,2, figsize = (8,8))



plt.suptitle('오후 오존 농도와 변수간의 관계', fontsize = 15)

ax1 = axes[0,0]
ax1.hist2d(df['오전 오존 농도'], df['오후 오존 농도'])
ax1.set_title('오전 오존 농도')
z = np.polyfit(df['오전 오존 농도'],df['오후 오존 농도'],deg=1)
p = np.poly1d(z)
ax1.plot(df['오전 오존 농도'], p(df['오전 오존 농도']), 'r-')
ax1.text(0.04, 0.02, f'y = {round(z[0],2)}x+{round(z[1],2)}', color = 'white', fontsize=15)


ax2 = axes[0,1]
ax2.hist2d(df['오전 이산화질소 농도 평균'], df['오후 오존 농도'])
ax2.set_title('오전 이산화질소 농도')
z = np.polyfit(df['오전 이산화질소 농도 평균'],df['오후 오존 농도'],deg=1)
p = np.poly1d(z)
ax2.plot(df['오전 이산화질소 농도 평균'], p(df['오전 이산화질소 농도 평균']), 'r-')
ax2.text(0.035, 0.02, f'y = {round(z[0],2)}x+{round(z[1],2)}', color = 'white', fontsize=15)

ax3 = axes[1,0]
ax3.hist2d(np.log(df['오전 기온 평균']), df['오후 오존 농도'])
ax3.set_title('오전 기온 평균')
z = np.polyfit(np.log(df['오전 기온 평균']),df['오후 오존 농도'],deg=1)
p = np.poly1d(z)
ax3.plot(np.log(df['오전 기온 평균']), p(np.log(df['오전 기온 평균'])), 'r-')
ax3.text(2.22, 0.1, f'y = {round(z[0],3)}x+{round(z[1],2)}', color = 'white', fontsize=15)


ax4 = axes[1,1]
ax4.hist2d(np.log(df['오전 습도 평균']), df['오후 오존 농도'])
ax4.set_title('오전 습도 평균')
z = np.polyfit(np.log(df['오전 습도 평균']),df['오후 오존 농도'],deg=1)
p = np.poly1d(z)
ax4.plot(np.log(df['오전 습도 평균']), p(np.log(df['오전 습도 평균'])), 'r-')
ax4.text(3.22, 0.02, f'y = {round(z[0],3)}x+{round(z[1],2)}', color = 'white', fontsize=15)

plt.tight_layout()
plt.savefig('Hist2D')