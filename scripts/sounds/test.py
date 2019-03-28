import librosa
import librosa.display as display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import manifold
from sklearn.decomposition import PCA
import skccm as ccm
from skccm.utilities import train_test_split


def load_data(sr=16000):
    filepath1 = '../../data/sounds/line_data/LINE1.m4a'
    filepath2 = '../../data/sounds/line_data/LINE3.m4a'
    filepath3 = '../../data/sounds/line_data/BASE.m4a'
    sound1, _ = librosa.load(filepath1, sr=sr)  # 183716-522043, 11.5-32.5
    sound2, _ = librosa.load(filepath2, sr=sr)
    base, _ = librosa.load(filepath3, sr=sr)
    # alignment
    # length = 129*sr
    length = min(len(sound1), len(sound2), len(base))
    base = base[6*sr: length+6*sr]
    return sound1, sound2, base


def check_crying(sound):
    # low amp last longer than 3 sec
    start, end = [], []
    temp = []
    flag = False
    for idx, amp in enumerate(sound):
        if amp <= 0.3:
            flag = True
            temp.append(idx)
        else:
            if flag:
                if len(temp) >= 5*16000:
                    start.append(temp[0])
                    end.append(temp[-1])
                temp = []
            else:
                pass
    return start, end


def causality():
    sr = 16000
    sound1, sound2, base = load_data(sr)
    # 12-33
    sound1 = sound1[12*sr:33*sr]
    base = base[12*sr:33*sr]
    embed = 2
    idx = 1
    for i in [1, 2, 4, 6, 8]:
        lag = i * sr
        e1 = ccm.Embed(sound1)
        e2 = ccm.Embed(base)
        X1 = e1.embed_vectors_1d(lag, embed)
        X2 = e2.embed_vectors_1d(lag, embed)
        x1tr, x1te, x2tr, x2te = train_test_split(X1, X2, percent=.75)
        CCM = ccm.CCM()
        len_tr = len(x1tr)
        lib_lens = np.arange(10, len_tr, len_tr/20, dtype='int')
        CCM.fit(x1tr, x2tr)
        x1p, x2p = CCM.predict(x1te, x2te, lib_lengths=lib_lens)
        sc1, sc2 = CCM.score()
        plt.subplot(5, 1, idx)
        # plt.plot(lib_lens, sc1)
        plt.plot(lib_lens, sc2,)
        idx += 1
    plt.show()


def audio_display():
    sr = 1000
    sound1, sound2, base = load_data(sr)
    plt.subplot(2, 1, 1)
    display.waveplot(sound1, sr=sr, alpha=0.25)
    plt.subplot(2, 1, 2)
    display.waveplot(base, sr=sr, alpha=0.25)
    plt.tight_layout()
    plt.show()


def state():
    sr = 16000
    sound1, sound2, base = load_data()
    # rearrange
    # 183716-522043 11~32
    cryset, baseset = [], []
    ts = []
    for i in range(12, 33):
        begin, end = i*sr, (i+1)*sr
        cryset.append(sound1[begin:end])
        baseset.append(base[begin:end])
        ts.append(i)
    cryset = np.array(cryset[2:])
    baseset = np.array(baseset[:-2])
    # tsne = manifold.TSNE(n_components=2, perplexity=5,
    #                      early_exaggeration=12, random_state=0)
    # tsnepos = tsne.fit_transform(graph)
    pca = PCA(n_components=1)
    xs = pca.fit_transform(baseset)
    print(pca.explained_variance_ratio_)
    print(pca.n_components_)
    ys = pca.fit_transform(cryset)
    print(pca.explained_variance_ratio_)
    print(pca.n_components_)
    plt.subplot(3, 1, 1)
    plt.scatter(ts[2:], ys)
    plt.xlabel('time')
    plt.ylabel('cry')

    plt.subplot(3, 1, 2)
    plt.scatter(ts[:-2], xs)
    plt.xlabel('time')
    plt.ylabel('music')

    plt.subplot(3, 1, 3)
    plt.scatter(xs, ys)
    plt.title('')
    plt.xlabel('music')
    plt.ylabel('cry')
    plt.show()


def temp():
    sr = 16000
    sound1, _, base = load_data()
    cryset, baseset = [], []
    ts = []
    for i in range(12, 33):
        begin, end = i*sr, (i+1)*sr
        cryset.append(sound1[begin:end])
        baseset.append(base[begin:end])
        ts.append(i)
    cryset = np.array(cryset[2:])
    baseset = np.array(baseset[:-2])
    # graph = []
    # for c, b in zip(cryset[2:], baseset[:-2]):
    #     g = c + b
    #     graph.append(g)
    # tsne = manifold.TSNE(n_components=2, perplexity=5,
    #                      early_exaggeration=12, random_state=0)
    # tsnepos = tsne.fit_transform(graph)
    # xs, ys = [], []
    # for d in tsnepos:
    #     xs.append(float(d[0]))
    #     ys.append(float(d[1]))
    plt.scatter(cryset, baseset, s=0.5)
    # plt.xlabel('time')
    # plt.ylabel('music')
    plt.show()


if __name__ == '__main__':
    # causality()
    state()
    # temp()
