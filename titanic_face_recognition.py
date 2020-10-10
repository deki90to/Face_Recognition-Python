import face_recognition
import cv2


video = cv2.VideoCapture('video/titanic/titanic.mp4')

jack_image = face_recognition.load_image_file('slike/titanic/jack.png')
jack_image_encoding = face_recognition.face_encodings(jack_image)[0]

kate_image = face_recognition.load_image_file('slike/titanic/kate.png')
kate_image_encoding = face_recognition.face_encodings(kate_image)[0]


known_face_encodings = [
	jack_image_encoding,
	kate_image_encoding,
]

known_face_names = [
	'Jack',
	'Kate',
]

while True:

	_, frame = video.read()
	rgb_frame = frame[:, :, ::-1]

	face_locations = face_recognition.face_locations(rgb_frame)
	face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)


	for (top, right, bottom, left), face_encoding in zip (face_locations, face_encodings):
		matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

		if True in matches:
			first_match_index = matches.index(True)
			name = known_face_names[first_match_index]
		else:
			name = 'Unknown'


		# name = 'Unknown'
		# face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
  		#best_match_index = np.argmin(face_distances)
  		#if matches[best_match_index]:
			#name = known_face_names[best_match_index]


		cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 1)
		# cv2.rectangle(frame, (left, bottom + 15), (right, bottom), (255, 0, 0), cv2.FILLED)

		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name, (left, bottom + 12), font, 0.5, (255, 255, 255), 1)


	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


video.release()
cv2.destroyAllWindows()