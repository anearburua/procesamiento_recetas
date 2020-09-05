exec(open("init_nlp.py").read())

# Obtiene una nueva estructura de datos Data Frame "df_token" después de tokenizar la columna
# "col" del Data Frame original "df".  
def tokenizar(df,col):
	data_token = []
	size = len(df['Receta'])
	
	# Obtener los tokens de todas las líneas de la columna
	for i in range(size):
		data_token.append(word_tokenize(df[col].values[i]))
	
	# Guardar los tokens en un nuevo Data Frame, junto al nombre de la receta 
	df_token = pd.DataFrame({'Receta':df['Receta'],'token': data_token})
	
	return df_token

# Asigna una etiqueta a cada elemento de la lista de tokens de todas las recetas. Guarda la nueva
# información añadiendo la columna "token_POS" al Data Frame "df".
def etiquedado(df):

	# Extraer el "tagger" de la carpeta
	
	if os.path.isfile('stanford-postagger-full-2017-06-09.zip'):
		extraer_tagger()

	# Definir la ubicación del archivo .jar y el modelo para el etiquetado
	jar = 'stanford-postagger-full-2017-06-09/stanford-postagger.jar'
	model = 'stanford-postagger-full-2017-06-09/models/spanish.tagger'
	
	# Definir el "tagger" de español
	spanish_tagger = StanfordPOSTagger(model, jar)

	tags = []
	size = len(df['Receta'])
	
	# Asignar una etiqueta a cada token de las recetas
	for i in range(size):
		tag = stanford_nltk(spanish_tagger.tag(df['token'].values[i]))
		tags.append(tag)
	
	# Guardar la información añadiendo una columna al Data Frame original
	df['token_POS'] = tags

	return df

# Extrae el etiquetador de Stanford	
def extraer_tagger():
	# Extraer el fichero al mismo directorio
	with zipfile.ZipFile('stanford-postagger-full-2017-06-09.zip', 'r') as zip_ref:
		zip_ref.extractall()
		
	# Borrar el archivo .zip
	os.remove('stanford-postagger-full-2017-06-09.zip')

# Convierte el etiquetado de Stanford POS tagger en etiquetado universal de nltk. Lee las tuplas 
# de pos_list, y crea la nueva lista new_pos_list	
def stanford_nltk(pos_list):
	# Definir excepciones que no etiqueta de forma correcta
	excep1 = ['champiñones', 'champiñon', 'pimienta', 'picadillo', 'bechamel', 'salsa', 'cucharada', 'cucharadas', 'zanahoria', 'zanahorias', 'puerro', 'puerros', 'patata', 'patatas', 'cebolleta', 'perejil', 'macarrones', 'bonito', 'gamba', 'gambas', 'piñones', 'chorizo', 'cucharada', 'lasaña', 'tomate', 'diente', 'anchoa', 'anchoas', 'nuez', 'nueces', 'langostino', 'langostinos', 'loncha', 'lonchas', 'alubia', 'alubias']
	excep2 = ['virgen']

	new_pos_list = []

	for i in range(len(pos_list)):
		tup = pos_list[i]
		tag = tup[1]
		
		# Definir la nueva etiqueta 'UNI' para unidades
		if ((tup[0]=='gramo')|(tup[0]=='gramos')|(tup[0]=='g')|(tup[0]=='gr')):
			t=('g','UNI')
		elif ((tup[0]=='kilo')|(tup[0]=='kilos')|(tup[0]=='kg')):
			t=('kg','UNI')
		elif ((tup[0]=='medio')|(tup[0]=='media')):
			t=(tup[0],'UNI')
			
		# Convertir la etiqueta de Standord en una comprensible para nltk
		else:
			if (tup[0] in excep1):
				new_tag = 'NOUN'
			elif (tup[0] in excep2):
				new_tag = 'ADJ'
			elif ((tag[0]=='z')|(tup[0]=='2')|(tup[0]=='3')|(tup[0]=='medio')):
				new_tag = 'NUM'
			elif (tag[0]=='a'):
				new_tag = 'ADJ'
			elif (tag[0]=='s'):
				new_tag = 'ADP'
			elif (tag[0]=='r'):
				new_tag = 'ADV'
			elif (tag[0]=='c'):
				new_tag = 'CONJ'
			elif (tag[0]=='d'):
				new_tag = 'DET'
			elif (tag[0]=='n'):
				new_tag = 'NOUN'
			elif (tag[0]=='p'):
				if ((tag[1]=='n')| (tag[1]=='i')):
					new_tag = 'NUM'
				else:
					new_tag = 'PRON'
			elif (tag[0]=='v'):
				new_tag = 'VERB'
			elif (tag[0]=='f'):
				new_tag = '.'
			else:
				new_tag = 'X'
			
			# Crear la tupla nueva
			t = (tup[0],new_tag)
			
		# Añadir la nueva tupla a la lista
		new_pos_list.append(t)

	return new_pos_list
			


# Define y separa los fragmentos del texto del Data Frame "df", dependiendo de la columna "col" 
# de la que se trate. Se crea una nueva columna 'chunks' con los fragmentos correspondientes 
def fragmentar(df,col):
	size = len(df['Receta'])
	recipes = []
	
	# Definir la gramática dependiendo de la columna de la que se trate
	if (col=='Ingredientes'):
		grammar = '''LAB:{(<NUM>|<DET>)(<NOUN>|<UNI>)((<ADP>(<NOUN>))?)*(<ADJ>(<ADP><NOUN>)?)?}
				{<NOUN><ADP><NOUN>}
				{<NOUN><ADJ>?}'''

	else:	
		grammar = '''LAB:{<VERB>(<NOUN>|<ADJ>|<ADP>|<ADV>|<CONJ>|<DET>|<NUM>|<PRON>|<.>|<X>)*}'''
	
	# Crear el parser para aplicar la gramática
	cp = nltk.RegexpParser(grammar) 

	# Aplicar la gramática, crear un árbol para cada receta, y consigue la lista de fragmentos
	for i in range(size):	
		tree = cp.parse(df['token_POS'].values[i]) 	
		#tree.draw()
		ingred = arbol_a_lista(tree)
		recipes.append(ingred)
	
	# Añadir una nueva columna para guardar fragmentos
	df['chunks'] = recipes

	return df

# Convierte el arbol "tree" que contiene los nodos con el texto fragmentado en la  lista ingred. 
# Cada lista devuelta pertenece a una receta
def arbol_a_lista(tree):
	ingred = []
	for node in tree:
	
		# Conseguir fragmentos del árbol
		if (type(node) is nltk.Tree):
			ingr = []
			
			# Conseguir fragmentos que coinciden con la gramática
			if node.label() == 'LAB':
				hojas = node.leaves()
				
				# Añadir el fragmento a una lista
				for i in range(len(hojas)):
					ingr.append(hojas[i][0])
					
			ingred.append(ingr)
			
	# Devolver las listas de fragmentos de una receta
	return ingred
	

# Preprocesa los fragmentos del Data Frame "df", eliminando los stopwords y obteniendo la raíz
# de las palabras. Crea un archivo .csv para guardar la información 
def preprocesar(df,col):
	size = len(df['Receta'])
	recipes = []
	recipes_clean = []
		
	for i in range(size):
		recipe = []
		recipe_clean = []
		token_list = df['chunks'].values[i]
		for j in range(len(token_list)):     
			tokens = []
			tokens_clean = []
			for z in range(len(token_list[j])):
			
				# Comprorar que la palabra no sea un stopword	
				if (token_list[j][z] not in spanish_stops):

					# Obtener la raíz
					steammed_word = spanish_stemmer.stem(token_list[j][z])
					if (steammed_word):
						tokens.append(steammed_word) 

					# Conseguir el etiquetado POS de la palabra
					pos = devolver_pos(i,df,token_list[j][z])
					
					# Obtener solo información valiosa
					if ((pos != 'NUM') and (pos != 'UNI') and (pos != 'DET') and (pos != 'ADJ')):
						if (steammed_word): tokens_clean.append(steammed_word)
			
			recipe.append(tokens)
			recipe_clean.append(tokens_clean)

		recipes.append(recipe)
		recipes_clean.append(recipe_clean)
		
	# Añadir dos nuevas columnas al Data Frame 
	df['pp'] = recipes
	df['pp_clean'] = recipes_clean
	
	# crear un archivo .csv
	df.to_csv(col+'_pp.csv')
	
	return df

# Dadas la palabra "word", el número de la receta "row" a la que pertenece, y la estructura Data
# frame, obtiene la etiqueta asignada a la palabra (en la columna 'token_POS') 
def devolver_pos(row,df,word):
	# Buscar la receta
	recipe = df['token_POS'].values[row]
	for i in range(len(recipe)):
		if (word==recipe[i][0]):
			# Devolver la etiqueta
			return recipe[i][1]


