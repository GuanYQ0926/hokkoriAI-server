import librosa
import librosa.display as display
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
from scipy.stats.stats import pearsonr
import json
# from pydub import AudioSegment


# acc -> mpf
# 0.001 sample 16k
def data_dump():
    base, sr = librosa.load('../../data/sounds/line_data/BASE.mp3', sr=1000)
    base = base[9*sr:]
    sound1, _ = librosa.load('../../data/sounds/line_data/LINE1.m4a', sr=1000)
    sound2, _ = librosa.load('../../data/sounds/line_data/LINE3.m4a', sr=1000)
    end = min(len(base), len(sound1), len(sound2))
    start, end = 1, end
    base, sound1, sound2 = base[start:end], sound1[start:end], \
        sound2[start:end]
    # derivative
    derivative1 = []
    delta = 0.001
    assert start >= 1
    for i in range(start, end-1):
        d = abs(sound1[i] - sound1[i-1]) / (abs(base[i] - base[i-1]) + delta)
        if d == 0.0 or d < 1.0:
            derivative1.append(d)
        else:
            e = math.log2(d)
            derivative1.append(e)
    # display
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(base[start:], sound1[start:], derivative1, c='b', marker='o')
    # ax.set_xlabel('song')
    # ax.set_ylabel('crying')
    # ax.set_zlabel('derivative')
    # plt.show()
    # dump
    song = base.tolist()[start: end-1]
    crying = sound1.tolist()[start: end-1]
    assert len(song) == len(derivative1)
    assert len(crying) == len(derivative1)
    jsonfile = dict(song=song, crying=crying, derivative=derivative1)
    with open('./data.json', 'w') as f:
        json.dump(jsonfile, f)


def audio_derivative():
    base, sr = librosa.load('../../data/sounds/line_data/BASE.m4a')
    base = base[9*sr:]
    sound1, _ = librosa.load('../../data/sounds/line_data/LINE1.m4a')
    sound2, _ = librosa.load('../../data/sounds/line_data/LINE3.m4a')
    length = min(len(base), len(sound1), len(sound2))
    base, sound1, sound2 = base[:length], sound1[:length], sound2[:length]
    derivative1, derivative2 = [], []
    delta = 0.0000001
    for i in range(1, length):
        derivative1.append((sound1[i] - sound1[i-1]) /
                           (base[i] - base[i-1] + delta))
        derivative2.append((sound2[i] - sound2[i-1]) /
                           (base[i] - base[i-1] + delta))
    fig, axarr = plt.subplots(2)
    axarr[0].set_title('crying/song')
    axarr[0].plot(derivative1, linewidth=0.5)
    axarr[1].plot(derivative2, linewidth=0.5)
    plt.show()


def audio_display():
    base, sr = librosa.load('../../data/sounds/line_data/BASE.m4a')
    sound1, _ = librosa.load('../../data/sounds/raw_sounds/crying_sounds/1.m4a')
    sound2, _ = librosa.load('../../data/sounds/line_data/LINE1.m4a')
    # length = min(len(base), len(sound1), len(sound2))
    # base, sound1, sound2 = base[:length], sound1[:length], sound2[:length]
    # fig, axarr = plt.subplots(3)
    # axarr[0].set_title('song & crying')
    # axarr[0].plot(base, linewidth=0.5)
    # axarr[1].plot(sound1, linewidth=0.5)
    # axarr[2].plot(sound2, linewidth=0.5)
    # plt.show()
    length = 6*sr
    base, sound1, sound2 = base[40*sr:length+40*sr], sound1[:length], sound2[50*sr:length+50*sr]
    plt.subplot(3, 1, 1)
    display.waveplot(base, sr=sr, alpha=0.25)
    plt.subplot(3, 1, 2)
    display.waveplot(sound1, sr=sr, alpha=0.25)
    plt.subplot(3, 1, 3)
    display.waveplot(sound2, sr=sr, alpha=0.25)
    plt.tight_layout()
    plt.show()


def smap(src, embed_E, k, tp, theta, delta_T=1):
    def least_squares(a, b):
        p, q = a.shape[0], a.shape[1]
        u, s, vh = np.linalg.svd(a, full_matrices=False)
        s_zeros = np.zeros([q, p])
        delta = 0.00001
        for i in range(q):
            if s[i] > delta * s[0]:
                s_zeros[i][i] = 1 / s[i]
        return np.matmul(np.matmul(np.matmul(vh, s_zeros), u), b)

    # attractor
    attractorX = []
    firstX = delta_T * (embed_E - 1)+1  # 1 index
    lastX = len(src)
    for i in range(firstX-1, lastX):
        attractorX.append(
            src[i-(embed_E-1)*delta_T: i+1: delta_T])
    attractorX = np.array(attractorX)
    predictY, observeY = [], []
    data = src[firstX-1:]
    # k nearest neighbors
    dlt = 0.00000001
    for veci, vecx in enumerate(attractorX):  # index & observation
        dists = [np.linalg.norm(vecx-vec)+dlt for vec in attractorX]
        indices = sorted(range(len(dists)), key=lambda i: dists[i])[:k+1]
        assert veci == indices[0]
        indices = indices[1:]
        ordered_dists = [dists[i] for i in indices]
        d1 = sum(ordered_dists) / float(k)
        # compute weight
        uis = [np.exp(-theta*di/d1) for di in ordered_dists]
        # reweighting matrix: shape k, k
        weight_matrix = []
        for i in range(k):
            temp = [0] * k
            temp[i] = uis[i]
            weight_matrix.append(temp)
        weight_matrix = np.array(weight_matrix)
        # design matrix: shape k, e+1
        assert embed_E >= 1
        a_matrix = []
        for indice in indices:
            a_matrix.append([1] + attractorX[indice])
        a_matrix = np.matmul(weight_matrix, np.array(a_matrix))
        # reponse vector: shape k, 1
        b_matrix = []
        for indice in indices:
            b_matrix.append([data[indice+tp]])
        b_matrix = np.matmul(weight_matrix, np.array(b_matrix))
        # least squares
        assert k >= embed_E + 1
        c = least_squares(weight_matrix, b_matrix)
        pred = c[0] + sum([c[i+1]*vecx[i] for i in range(embed_E)])
        predictY.append(pred)
        observeY.append(src[veci+tp])


def simplex_projection(src, embed_E, advance=1, delta_T=1):
    k = embed_E - 1
    # attractor
    attractorX = []
    firstX = delta_T * (embed_E - 1)+1  # 1 index
    lastX = len(src)
    for i in range(firstX-1, lastX):
        attractorX.append(
            src[i-(embed_E-1)*delta_T: i+1: delta_T])
    attractorX = np.array(attractorX)
    test_data = src[firstX-1:]
    # k nearest neighbors
    # distances & sort & predict
    dlt = 0.00000001
    predictY, observeY = [], []
    for veci, vecx in enumerate(attractorX):  # index & observation
        if veci == len(test_data) - advance:
            break
        dists = [np.linalg.norm(vecx-vec)+dlt for vec in attractorX]
        indices = sorted(range(len(dists)), key=lambda i: dists[i])[:k+1]
        assert veci == indices[0]
        indices = indices[1:]
        ordered_dists = [dists[i] for i in indices]
        d1 = ordered_dists[0]
        uis = [np.exp(-di/d1) for di in ordered_dists]
        N = sum(uis)
        ws = [ui / N for ui in uis]
        pred = 0
        for wi, ind in zip(ws, indices):
            if ind+advance >= len(test_data) - advance:
                continue
            pred += wi*test_data[ind+advance]
        predictY.append(pred/sum(ws))
        observeY.append(test_data[veci+advance])
    assert len(predictY) == len(observeY)
    p, _ = pearsonr(np.array(predictY), np.array(observeY))
    return p


if __name__ == '__main__':
    # data_dump()
    # audio_display()
    audio_derivative()
