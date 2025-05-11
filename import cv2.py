import cv2
import requests
import time
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

# Configuración de Clarifai (necesitarás una API key)
CLARIFAI_API_KEY = 'Inserte API key'  # Reemplaza con tu API key de Clarifai
USER_ID = '2jl42efzat1y'
APP_ID = 'Reconocimiento-de-personas'
MODEL_ID = 'general-image-recognition'
MODEL_VERSION_ID = 'aa7f35c01e0642fda5cf400f543e7c40'

def tomar_foto():
    # Inicializar la cámara
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return None
    
    print("Presiona 's' para tomar la foto o 'q' para salir")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame")
            break
            
        cv2.imshow('Cámara - Presiona "s" para tomar foto', frame)
        
        key = cv2.waitKey(1)
        if key == ord('s'):
            # Guardar la imagen temporalmente
            filename = 'foto_capturada.jpg'
            cv2.imwrite(filename, frame)
            print(f"Foto guardada como {filename}")
            break
        elif key == ord('q'):
            print("Saliendo sin tomar foto")
            filename = None
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    return filename

def analizar_imagen_con_clarifai(image_path):
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    
    metadata = (('authorization', f'Key {CLARIFAI_API_KEY}'),)
    
    with open(image_path, 'rb') as f:
        file_bytes = f.read()
    
    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID),
        model_id=MODEL_ID,
        version_id=MODEL_VERSION_ID,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(
                        base64=file_bytes
                    )
                )
            )
        ])
    
    response = stub.PostModelOutputs(request, metadata=metadata)
    
    if response.status.code != status_code_pb2.SUCCESS:
        print(f"Error en la solicitud: {response.status.description}")
        return None
    
    return response.outputs[0].data.concepts

def interpretar_resultados(concepts):
    es_persona = False
    otros_objetos = []
    conceptos_persona = ['person', 'people', 'adult', 'man', 'woman', 'child', 'portrait']
    
    for concept in concepts:
        if concept.name in conceptos_persona and concept.value > 0.85:
            es_persona = True
        elif concept.value > 0.7:
            otros_objetos.append((concept.name, concept.value))
    
    otros_objetos.sort(key=lambda x: x[1], reverse=True)
    return es_persona, otros_objetos[:5]

def main():
    # Paso 1: Tomar la foto
    imagen = tomar_foto()
    if not imagen:
        return
    
    # Paso 2: Analizar la imagen con Clarifai
    print("Analizando imagen...")
    conceptos = analizar_imagen_con_clarifai(imagen)
    
    if not conceptos:
        print("No se pudo analizar la imagen")
        return
    
    # Paso 3: Interpretar los resultados
    es_persona, otros_objetos = interpretar_resultados(conceptos)
    
    # Paso 4: Mostrar resultados
    if es_persona:
        print("\n¡Se ha detectado una persona en la imagen!")
    else:
        print("\nNo se detectó una persona en la imagen. Los objetos identificados son:")
        for obj, confianza in otros_objetos:
            print(f"- {obj} (confianza: {confianza:.2%})")

if __name__ == "__main__":
    main()