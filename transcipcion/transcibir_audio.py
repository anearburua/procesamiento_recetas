exec(open("init_transcripcion.py").read())

# Transcribe un único archivo de audio, dadas la ubicación "storage_uri", la duración "duration" y la ubicación junto al nombre "transcription_path" del nuevo archivo que se transcribirá 
def reconocer_audio(storage_uri,duration,transcription_path):

	# Acceder a los servicios de Speech-to-Text
	client = speech_v1.SpeechClient()

	# Hercios de archivos de audio .wav
	sample_rate_hertz = 44100

	# Lenguaje del audio
	language_code = "es-ES"

	# Encoding de los datos de audio enviados 
	encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16
	
	config = {
	"sample_rate_hertz": sample_rate_hertz,
	"language_code": language_code,
	"encoding": encoding,
	}
	audio = {"uri": storage_uri}

	# Hacer la llamada de transcripción según duración del audio
	if duration<60.0:
		response = client.recognize(config, audio)
	else:
		operation = client.long_running_recognize(config, audio)
		response = operation.result()

	# Escribir resultado en el nuevo fichero .txt
	for result in response.results:
	
		# La primera  alternativa es el resultado mas probable
		alternative = result.alternatives[0]
		f = open(transcription_path, "a")
		f.write(format(alternative.transcript))
		f.close()

