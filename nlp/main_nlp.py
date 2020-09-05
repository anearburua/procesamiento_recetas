from process import *
from train import *
from consultations import *

exec(open("init_nlp.py").read())

# Procesa las recetas e inicializa el menú para extraer información de ellas
def nlp_principal():

	ingr = 'Ingredientes'
	elab = 'Elaboración'
	df = cargar_ficheros()

	# Procesar la columna de los ingredientes
	print("Procesando los ingredientes...")
	df_token_ingr = tokenizar(df,ingr)
	df_tag_ingr = etiquedado(df_token_ingr)
	df_chunk_ingr = fragmentar(df_tag_ingr,ingr)
	df_pp_ingr = preprocesar(df_chunk_ingr,ingr)
	corpus_ingr = obtener_corpus(df_pp_ingr)
	model_ingr = entrenar(corpus_ingr,ingr)
	
	# Procesar la columna de las elaboraciones
	print("Procesando las elaboraciones...")
	df_token_elab = tokenizar(df,elab)
	df_tag_elab = etiquedado(df_token_elab)
	df_chunk_elab = fragmentar(df_tag_elab,elab)
	df_pp_elab = preprocesar(df_chunk_elab,elab)
	corpus_elab = obtener_corpus(df_pp_elab)
	model_elab = entrenar(corpus_elab,elab)

	# Iniciar el menu de consultas
	menu(df,df_pp_ingr,corpus_ingr,model_ingr,df_pp_elab,corpus_elab,model_elab)

	
# Carga los ficheros de la carpeta "transcripciones" y crea el Data Frame "df" para guardar el
# texto de todos los archivos. Crea el documento .csv "recetas.csv" con la misma información
def cargar_ficheros():
	print("Cargando ficheros...")
	
	# Cargar todas las transcripciones 
	d = os.path.dirname(os.getcwd())
	direc = os.path.join(d,"transcripcion","transcripciones")
	names = os.listdir(direc)
	sorted_names = sorted(names)

	data_names = []
	data_ingr = []
	data_elab = []

	# Obtener la informacion de las transcripciones
	for file in sorted_names:
		fname = os.path.join(direc,file)
		with open(fname,'r') as f:
			d1 = f.read().lower().split("ingredientes")
			d2 = d1[1].split("elaboración")
			data_names.append(d1[0])
			data_ingr.append(d2[0])
			data_elab.append(d2[1])
	# Crear la estructura de datos y el archivo csv
	df = pd.DataFrame({'Receta': data_names, 'Ingredientes': data_ingr, 'Elaboración': data_elab})

	df.to_csv('recetas.csv')
	
	return df

# Crea el menú de opciones con la información de las recetas procesadas	
def menu(df,df_pp_ingr,corpus_ingr,model_ingr, df_pp_elab,corpus_elab,model_elab):
	end = False
	
	# Imprimir el menú
	while not end:
		print("\n ---------------------\n","| Opciones posibles |\n","---------------------\n")
		print("1. Receta con elaboración similar")
		print("2. Receta con ingredientes similares")
		print("3. Recetas posibles")
		print("4. Utilidad de ingrediente")
		print("5. Ver recetas")
		print("6. Salir")
		
		# Leer la opción elegida
		opcion = elegir_opcion()
		
		# Llamar a funciones necesarias según opción
		if opcion == 1:
			recipe = elegir_receta(df)
			elaboracion_similar(df,recipe,model_elab,corpus_elab)		
		elif opcion == 2:
			recipe = elegir_receta(df)
			ingrediente_similar(df,recipe,model_ingr,corpus_ingr)
		elif opcion == 3:
			ingred = elegir_ingrediente(opcion)
			recetas_posibles(ingred,df_pp_ingr)
		elif opcion == 4:
			ingred = elegir_ingrediente(opcion)
			utilidad_ingrediente(ingred,df_pp_elab)
		elif opcion == 5:
			ver_recetas(df)
		elif opcion == 6:
			end = True
		else:
			print('Introduce un número del 1 al 6')

# Obtiene la opción elegida por el usuario a través de la terminal	
def elegir_opcion():
	valid = False
	while not valid:
		try:
			num = int(input("\nElige una opción: "))
			print("---------------------\n")
			valid = True
		except ValueError:
			print('Introduce un número')
	return num

# Obtiene el número de la receta elegida por el usuario	
def elegir_receta(df):
	valid = False
	while not valid:
		try:
			n = int(input("Introduce el número de la receta: "))
			
			# El número no puede ser mayor que la cantidad de recetas
			if (n<=len(df['Receta'])):
				valid = True
			else: 
				print("El número de la receta no es válido\n")
		except ValueError:
			print("Introduce un número\n")
	return n

# Obtiene el ingrediente o los ingredientes elegidos por el usuario				
def elegir_ingrediente(op):
	valid = False
	while not valid:
		# Uno o más ingredientes
		if (op == 3):
			i = input("Introduce un ingrediente, o varios separados por comas: ")
			print("\n")
			valid = True
			return list(i.split(","))
		# Solo un ingrediente
		else:
			i = input("Introduce un ingrediente: ")
			print("\n")
			valid = True
			return list(i.split(","))
			
# Imprime los nombres de las recetas del Data Frame "df"
def ver_recetas(df):
	for i in range(len(df['Receta'])):
		print(i,". ", df['Receta'].values[i])
		
												

# Llamada al programa principal
if __name__ == "__main__":
    	nlp_principal()
    	





	
