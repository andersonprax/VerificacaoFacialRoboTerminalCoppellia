import cv2
import os
import numpy as np

# Rota de armazenamento de dados
dataPath = "C:/Users/aps8/Documents/VerificacaoFacialRoboTerminalCoppellia/Data" # Mudar a rota onde esta armazenada a data
peopleList = os.listdir(dataPath)
print('Lista de pessoas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
	personPath = dataPath + '/' + nameDir
	print('Lendo as imagens')

	for fileName in os.listdir(personPath):
		print('Rostos: ', nameDir + '/' + fileName)
		labels.append(label)
		facesData.append(cv2.imread(personPath+'/'+fileName,0))

	label = label + 1

# Métodos para treinar o reconhecedor
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Treinando o reconhecedor de rostos
print("Treinando...")
face_recognizer.train(facesData, np.array(labels))

# Armazenando o modelo obtido
face_recognizer.write('modeloLBPHFace.xml')
print("Modelo armazenado...")