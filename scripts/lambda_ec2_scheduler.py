# Lambda function code for EC2 start/stop scheduler
# Package this as lambda_ec2_scheduler.zip

import boto3
import os

ec2 = boto3.client('ec2')
INSTANCE_ID = os.environ['INSTANCE_ID']

def start_handler(event, context):
    """Start the EC2 instance"""
    try:
        ec2.start_instances(InstanceIds=[INSTANCE_ID])
        print(f"✅ Started EC2 instance: {INSTANCE_ID}")
        return {
            'statusCode': 200,
            'body': f'Started instance {INSTANCE_ID}'
        }
    except Exception as e:
        print(f"❌ Error starting instance: {str(e)}")
        raise

def stop_handler(event, context):
    """Stop the EC2 instance"""
    try:
        ec2.stop_instances(InstanceIds=[INSTANCE_ID])
        print(f"✅ Stopped EC2 instance: {INSTANCE_ID}")
        return {
            'statusCode': 200,
            'body': f'Stopped instance {INSTANCE_ID}'
        }
    except Exception as e:
        print(f"❌ Error stopping instance: {str(e)}")
        raise
