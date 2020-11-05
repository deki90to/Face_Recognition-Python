import face_recognition
import cv2

# Starting web cam (0), or connected device (1), or path to local video
video = cv2.VideoCapture('video/newkids/newkids.mp4')
# video = cv2.VideoCapture(1)

# Loading pictures and encoding faces(learn how to recognize it)
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


# Creating arrays of known face encodings
known_face_encodings = [
	barrie_image_encoding,
	gerrie_image_encoding,
	richard_image_encoding,
	rikket_image_encoding,
	robbie_image_encoding,
]

# Creating arrays of known face names
known_face_names = [
	'Barrie',
	'Gerrie',
	'Richard',
	'Rikket',
	'Robbie',
]

# Choosing codecs and saving output video to 'output.mp4'
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# 20.0 - frames
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (800, 600))

while True:
	# Getting every frame in video
	ret, frame = video.read()
	# less_frames = cv2.resize(frame, (800, 600), fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
	# rgb_frame = less_frames[:, :, ::-1]

	# Converting BGR to RGB color that face recognition using
	rgb_frame = frame[:, :, ::-1]

	# Find faces in frame
	face_locations = face_recognition.face_locations(rgb_frame)

	# Find encodings in frame
	face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

	# loop through faces in video frame 
	for (top, right, bottom, left), face_encoding in zip (face_locations, face_encodings):

		# Check if face match known faces
		matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

		# if face is found in known face encodings, use first one
		if True in matches:
			first_match_index = matches.index(True)
			name = known_face_names[first_match_index]

		# If no faces found, it's Unknown
		else:
			name = 'Unknown'

		# Drow a rectangle around the face
		cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 1)

		# Draw label under the rectangle that contain text (name)
		# cv2.rectangle(frame, (left, bottom + 15), (right, bottom), (255, 0, 0), cv2.FILLED)

		# Font
		font = cv2.FONT_HERSHEY_DUPLEX

		# Adjusting text inside the label
		cv2.putText(frame, name, (left, bottom + 12), font, 0.5, (255, 255, 255), 1)

		# While return is True, frames will be set to 'out' instance
		if ret == True:
			out.write(frame)

	# Display result
	cv2.imshow('Video', frame)

	# Press 'Q' to quit from running video
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release webcam
out.release()
video.release()
cv2.destroyAllWindows()