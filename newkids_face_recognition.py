import face_recognition
import cv2


video = cv2.VideoCapture('video/newkids/newkids.mp4')

barrie_image = face_recognition.load_image_file('slike/newkids/Barrie.png')
barrie_image_encoding = face_recognition.face_encodings(barrie_image)[0]

gerrie_image = face_recognition.load_image_file('slike/newkids/Gerrie.png')
gerrie_image_encoding = face_recognition.face_encodings(gerrie_image)[0]

richard_image = face_recognition.load_image_file('slike/newkids/Richard.png')
richard_image_encoding = face_recognition.face_encodings(richard_image)[0]

rikket_image = face_recognition.load_image_file('slike/newkids/Rikket.png')
rikket_image_encoding = face_recognition.face_encodings(rikket_image)[0]

robbie_image = face_recognition.load_image_file('slike/newkids/Robbie.png')
robbie_image_encoding = face_recognition.face_encodings(robbie_image)[0]


known_face_encodings = [
	barrie_image_encoding,
	gerrie_image_encoding,
	richard_image_encoding,
	rikket_image_encoding,
	robbie_image_encoding,
]

known_face_names = [
	'Barrie',
	'Gerrie',
	'Richard',
	'Rikket',
	'Robbie',
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