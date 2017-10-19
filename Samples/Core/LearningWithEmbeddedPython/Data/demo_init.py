#***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#**************************************************************************
from tensorflow.contrib.keras.api.keras.layers import Input
from tensorflow.contrib.keras.api.keras.layers import UpSampling2D, UpSampling3D, Reshape
from tensorflow.contrib.keras.api.keras.models import Model, Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras.layers import Dropout
from tensorflow.contrib.keras.api.keras.layers import Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv1D, Conv2D
from tensorflow.contrib.keras.api.keras.layers import MaxPooling2D
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras import backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
K.set_image_data_format('channels_last')
np.random.seed(7)

def CreateSimpleImageModel_128():
	dataIn = Input(shape=(3,))
	layer = Dense(4 * 4, activation='tanh')(dataIn)
	layer = Dense(128 * 128 * 4, activation='linear')(layer)
	layer = Reshape((128, 128, 4))(layer)
	layer = UpSampling3D((4, 4, 1))(layer)
	layer = Reshape((1, 512, 512, 4))(layer)
	modelOut = layer
	model = Model(inputs=[dataIn], outputs=[modelOut])
	adam = Adam(lr=0.005, decay=0.0001)
	model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
	return model

def CreateSimpleImageModel_256():
	dataIn = Input(shape=(3,))
	layer = Dense(4 * 4, activation='tanh')(dataIn)
	layer = Dense(256 * 256 * 4, activation='linear')(layer)
	layer = Reshape((256, 256, 4))(layer)
	layer = UpSampling3D((2, 2, 1))(layer)
	layer = Reshape((1, 512, 512, 4))(layer)
	modelOut = layer
	model = Model(inputs=[dataIn], outputs=[modelOut])
	adam = Adam(lr=0.005, decay=0.0001)
	model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
	return model

def CreateSimpleImageModel_512():
	dataIn = Input(shape=(3,))
	layer = Dense(4 * 4, activation='tanh')(dataIn)
	layer = Dense(512 * 512 * 4, activation='linear')(layer)
	layer = Reshape((1, 512, 512, 4))(layer)
	modelOut = layer
	model = Model(inputs=[dataIn], outputs=[modelOut])
	adam = Adam(lr=0.005, decay=0.0001)
	model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
	return model

def ConvertDataToNumpy( trainDataIn, trainDataOut, resOutW, resOutH ):
	npInput = np.ones((1, trainDataIn.shape[0]), dtype='float32')
	npInput[0] = trainDataIn
	tmpData = trainDataOut.reshape(resOutW,resOutH,4) / 256.0
	npOutput = np.zeros((1, 1, resOutW, resOutH,4), dtype='float32')
	npOutput[0][0] = np.array(tmpData, dtype='float32')
	return (npInput, npOutput)