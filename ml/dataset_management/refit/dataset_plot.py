"""Plots normalized dataset"""

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIRECTORY = '/home/lindo/Develop/nilm/ml/dataset_management/refit/'
APPLIANCE_NAME = 'washingmachine'
FILE_NAME = 'washingmachine_test_H8.csv'

CHUNK_SIZE = 10 ** 6

for idx, chunk in enumerate(pd.read_csv(DATA_DIRECTORY + APPLIANCE_NAME + '/' + FILE_NAME,
                                        names=['aggregate', APPLIANCE_NAME, 'status'],
                                        # iterator=True,
                                        # #skiprows=15 * 10 ** 6,
                                        chunksize=CHUNK_SIZE,
                                        header=0)):

    fig = plt.figure(num='Figure {:}'.format(idx))
    ax1 = fig.add_subplot(111)

    ax1.plot(chunk['aggregate'])
    ax1.plot(chunk[APPLIANCE_NAME])
    ax1.plot(chunk['status'])

    ax1.grid()
    ax1.set_title('{:}'.format(FILE_NAME), fontsize=14, fontweight='bold')
    ax1.set_ylabel('Power normalized')
    ax1.set_xlabel('samples')
    ax1.legend(['aggregate', APPLIANCE_NAME, 'status'])

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

    del chunk