from google.oauth2 import service_account
from google.cloud import vision
import cv2
import io

def draw_ocr_results(image, text, rect, color=(0, 255, 0)):
	# unpacking the bounding box rectangle and draw a bounding box
	# surrounding the text along with the OCR'd text itself
	(startX, startY, endX, endY) = rect
	cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
	cv2.putText(image, text, (startX, startY - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
	# return the output image
	return image

# create the client interface to access the Google Cloud Vision API
credentials = service_account.Credentials.from_service_account_file(filename='path to your_service_account_key.json',
																	scopes=["https://www.googleapis.com/auth/cloud-platform"])
client = vision.ImageAnnotatorClient(credentials=credentials)
# load the input image as a raw binary file (this file will be
# submitted to the Google Cloud Vision API)



with io.open('images/image.png', "rb") as f:
	byteImage = f.read()
	

# create an image object from the binary file and then make a request
# to the Google Cloud Vision API to OCR the input image
print("[INFO] making request to Google Cloud Vision API...")
image = vision.Image(content=byteImage)
response = client.text_detection(image=image)
# check to see if there was an error when making a request to the API
if response.error.message:
	raise Exception(
		"{}\nFor more info on errors, check:\n"
		"https://cloud.google.com/apis/design/errors".format(
			response.error.message))


# read the image again, this time in OpenCV format and make a copy of
# the input image for final output
image = cv2.imread('images/image.png')
final = image.copy()
# loop over the Google Cloud Vision API OCR results
for text in response.text_annotations[1::]:
	# grab the OCR'd text and extract the bounding box coordinates of
	# the text region
	ocr = text.description
	startX = text.bounding_poly.vertices[0].x
	startY = text.bounding_poly.vertices[0].y
	endX = text.bounding_poly.vertices[1].x
	endY = text.bounding_poly.vertices[2].y
	# construct a bounding box rectangle from the box coordinates
	rect = (startX, startY, endX, endY)
	


    	# draw the output OCR line-by-line
	output = image.copy()
	output = draw_ocr_results(output, ocr, rect)
	final = draw_ocr_results(final, ocr, rect)
	# show the output OCR'd line
	print(ocr)
	cv2.imshow("Output", output)
	cv2.waitKey(0)
# show the final output image
cv2.imshow("Final Output", final)
cv2.waitKey(0)





