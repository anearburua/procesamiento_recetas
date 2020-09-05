exec(open("init_nlp.py").read())

# A partir del último Data Frame "df", crea la lista "recipe_list" con el corpus definitivo
# para el aprendizaje del modelo
def obtener_corpus(df):
	list_ingr = df['pp_clean']
	count_ingr = 0
	all_ingr = []
	recipe_list = []
	for i in range(len(list_ingr)):
		recipe_ingr = []
		for j in range(len(list_ingr[i])):
		
			# Contar el número de ingredientes totales
			count_ingr+=1
			
			for z in range(len(list_ingr[i][j])):
			
				# Añadir cada token a la lista de vocabulario
				all_ingr.append(list_ingr[i][j][z])
				recipe_ingr.append(list_ingr[i][j][z])
				
		# Crear la lista de vocabulario para cada receta
		recipe_list.append(recipe_ingr)
			
	# Obtener el ingredientes más comunes (de todas las recetas) 
	#num_aparicion(all_ingr)
	
	return recipe_list
		

# Obtiene los ingredientes y los pares de ingredientes más comunes de todas las recetas conjuntas
# guardadas en "ingr"
def num_aparicion(ingr):
	# Ingredientes más comunes
	ingrFreq = collections.Counter(ingr)
	ingr_common = ingrFreq.most_common(50)

	# Pares de ingredientes (2-gram) más comunes
	bigram = ngrams(ingr,2)
	bigramFreq = collections.Counter(bigram)
	bigram_common = bigramFreq.most_common(10)

	# Imprimir resultados
	print('Palabras más comunes: ', ingr_common)
	print('2-gram más comunes: ', bigram_common)
	
	# Dibujar gráfico con los resultados
	fd = nltk.FreqDist(ingr)
	fd.plot(30,cumulative=False)
	
# Dada la lista con el corpus "cor", define los parámetros de la red neuronal, y entrena el 
# vocabulario para crear y devolver el modelo "model"
def entrenar(cor,name):
	# Definir los parámetros del modelo
	num_features = 100                        
	min_word_count = 2                       
	num_workers = 4      
	context = 10 
          
        # Crear el modelo con el corpus y la configuración  
	model = word2vec.Word2Vec(cor,workers=num_workers, size=num_features, sg = 1, min_count = min_word_count, window = context, iter=20)

	# Obtener vocabulario y vectores 
	vocab = list(model.wv.vocab)
	X = model[vocab]
	
	# Dibujar vocabulario
	#dibujar(X,vocab)
	
	# Guardar y devolver el modelo creado y entrenado
	model.save(name+'_modelo.w2v')
	
	return model
	
# Dibujar los vectores "vector" con las etiquetas "labels"	
def dibujar(vector,labels):
	
	# Reducir dimensión de vectores a 2
	tsne = TSNE(n_components=2)
	X_2d = tsne.fit_transform(vector)
	
	# Definir coordenadas
	x_coords = X_2d[:, 0]
	y_coords = X_2d[:, 1]
	
	# Dibujar gráfico
	plt.scatter(x_coords, y_coords)

	# Asignar etiqueta a cada punto
	for label, x, y in zip(labels, x_coords, y_coords):
		plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
		plt.xlim(x_coords.min()-50, x_coords.max()+50)
		plt.ylim(y_coords.min()-50, y_coords.max()+50)
		
	plt.show()

