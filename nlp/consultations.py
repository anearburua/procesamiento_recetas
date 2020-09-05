exec(open("init_nlp.py").read())
from train import *

# Dada una la receta "recipe", busca la receta con la elaboración más similar 
def elaboracion_similar(df,recipe,model,recipe_list):
	similar_recipe, similar_recipe_similarity = buscar_similar(df,recipe,model,recipe_list)
	print("\nReceta introducida: ", df['Receta'].values[recipe] )
	print("Receta con elaboración similar: ", similar_recipe )
	print("Similitud: ", similar_recipe_similarity )
	

# Dada la receta "recipe", busca la receta con ingredientes más similares 
def ingrediente_similar(df,recipe,model,recipe_list):
	similar_recipe, similar_recipe_similarity = buscar_similar(df,recipe,model,recipe_list)
	print("\nReceta introducida: ", df['Receta'].values[recipe] )
	print("Receta con ingredientes similares: ", similar_recipe )
	print("Similitud: ", similar_recipe_similarity )


# Busca la receta mas similar a "recipe", con el modelo "model" entrenado con el corpus "recipe_list"
def buscar_similar(df,recipe,model,recipe_list):
	
	# calcular vectores de todas las recetas
	vector = embeed_vector(model,recipe_list)
	
	# Obtener lista de distancias ordenada 
	dist, order = cosine_dist(vector,recipe)
	
	# Obtener receta más similar y similitud
	similar_recipe = df['Receta'].values[order[1]]
	similar_recipe_similarity = dist[1]
	
	# Dibujar vectores
	#labels = list(range(0,len(vector)+1))
	#dibujar(vector,labels)

	return similar_recipe, similar_recipe_similarity

# Crea un vector por cada receta que se encuentre en "recipe_list", con los valores del vocabulario
# del modelo "model"
def embeed_vector(model,recipe_list):
	vec = []
	for i in range(len(recipe_list)):
		words = [word for word in recipe_list[i] if word in model.wv.vocab]
		vec.append(np.mean(model.wv[words], axis=0))
	return vec

# Consigre la lista "l1" de las similitudes de coseno de todos los vectores de recetas "v" frente a 
# la receta número "num". "l2" es el orden del número de las recetas de más similar a menos		
def cosine_dist(v,num):
	cos_list = []
	num_list = list(range(0,len(v)+1))
	
	# Calcular la matriz de similitud de coseno
	cos_matrix = cosine_similarity(v, v)
	
	# Crear un heatmap con la matriz, y dibujarlo
	#ax = sns.heatmap(cos_matrix)
	#plt.show()
	
	for i in range(len(v)):
		for j in range(i,len(v)):
		
			# Obtener las similitudes con la receta número "num"
			if (j==num or i==num):
				cos_list.append(cos_matrix[i][j])
				
	# Ordenar la lista de similitud de mayor a menor
	l1, l2 = zip(*sorted(zip(cos_list,num_list),reverse=True))
	return list(l1), list(l2)
			
		
# Dada la lista de ingredientes "ingred", consigue las recetas elaborables con esos ingredientes
def recetas_posibles(ingred,df_pp):
	ingred_pp = []
	
	# Preprocesar la lista de ingredientes
	for i in range(len(ingred)):
		if (ingred[i] not in spanish_stops):
			steammed_word = spanish_stemmer.stem(ingred[i])
			if (steammed_word): ingred_pp.append(steammed_word)
	
	# Buscar recetas posibles con sus correspondientes ingredientes buscados	
	recipe,coin = buscar_ingredientes(ingred_pp, df_pp)
	
	# Imprimir información obtenida
	if recipe:
		for i in range(len(recipe)):
			print("La receta ", df_pp['Receta'].values[recipe[i]], "contiene los ingredientes: ")
			for j in range(len(coin[i])):
				print("-> "," ".join(coin[i][j]))

	else:
		print("No se ha encontrado ninguna receta con esos ingredientes")


# Dada la lista de ingredientes preprocesada "ingred_pp", busca las recetas posibles en la 
# estructura "df_pp"
def buscar_ingredientes(ingred_pp,df_pp):
	recipes_pp = df_pp['pp_clean']
	recipe = []
	coincidences = []
	
	# Analizar cada receta preprocesada
	for i in range(len(recipes_pp)):
		count=0
		ingr=[]
		for j in range(len(recipes_pp[i])):
			for z in range(len(recipes_pp[i][j])):
				
				# Obtener ingrediente buscado (con toda información, no preprocesado)
				if (recipes_pp[i][j][z] in ingred_pp):
					ingr.append(df_pp['chunks'].values[i][j])
					count+=1
					
		# Si una receta contiene todos los ingredientes, añadir a la lista		
		if(count==len(ingred_pp)):
			recipe.append(i)
			coincidences.append(ingr)
	
	return recipe, coincidences
				
				

# Explica como utilizar el ingrediente "ingred" en las elaboraciónes
def utilidad_ingrediente(ingred,df_pp):

	# Preprocesa el ingrediente
	if (ingred[0] not in spanish_stops):
		steammed_ingr = spanish_stemmer.stem(ingred[0])
	else:
		print("El ingrediente no es válido")
	
	# Buscar las recetas con las elaboraciones que contengan el ingrediente
	recipe, coin = buscar_elaboraciones([steammed_ingr],df_pp)
	
	# Imprimir información obtenida
	if recipe:
		for i in range(len(recipe)):
			print("La receta ", df_pp['Receta'].values[recipe[i]], "utiliza el ingrediente ", ingred[0], ":")
			for j in range(len(coin[i])):
				print("-> "," ".join(coin[i][j]))
	else:
		print("No se ha encontrado ninguna elaboración con ese ingrediente")

						
# Busca las partes de las elaboraciones en "df_pp", que contengan el ingrediente "ingred_pp"
def buscar_elaboraciones(ingred_pp,df_pp):
	recipes_pp = df_pp['pp_clean']
	recipe = []
	coincidences = []
	
	# Analizar todas las elaboraciones preprocesadas
	for i in range(len(recipes_pp)):
		ingr=[]
		for j in range(len(recipes_pp[i])):
			for z in range(len(recipes_pp[i][j])):
			
				# Obtener elaboración (con toda información, antes del preprocesado)
				if (recipes_pp[i][j][z] in ingred_pp):
					ingr.append(df_pp['chunks'].values[i][j])

		# Añadir nombre de la receta y parte de la elaboración correspondiente a la lista
		if(ingr):
			coincidences.append(ingr)
			recipe.append(i)
	
	return recipe, coincidences
	


