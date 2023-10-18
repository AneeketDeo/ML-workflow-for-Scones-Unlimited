# 1. datageneration.py

import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']

    # Download the data from s3 to /tmp/image.png
    s3.download_file(bucket, key, '/tmp/image.png')
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        "image_data": image_data,
        "s3_bucket": bucket,
        "s3_key": key,
        "inferences": []
    }

###########################################################################

# 2. classification.py

import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2023-10-14-06-57-03-036"

def lambda_handler(event, context):

    # Decode the image data
    image_data = event.get('image_data', '')

    image = base64.b64decode(image_data)

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(endpoint_name=ENDPOINT)

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction:
    inferences = predictor.predict(image)

    # We return the data back to the Step Function    
    event["inferences"] = inferences.decode('utf-8')
    return event

###########################################################################

# 3. threshold.py

import json

THRESHOLD = .93

def lambda_handler(event, context):
    
    inferences_str = event.get('inferences', '')
    
     # Check if any values in our inferences are above THRESHOLD
    inferences = [float(value) for value in inferences_str.strip('[]').split(', ')]

    meets_threshold = any(value > THRESHOLD for value in inferences)

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise Exception ("THRESHOLD_CONFIDENCE_NOT_MET")



    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }