#!/usr/bin/env groovy

@Grapes([
    @Grab(group='org.apache.tika', module='tika-core', version='1.18'),
    @Grab(group='com.amazonaws', module='aws-java-sdk-rekognition', version='1.11.408'),
    @Grab(group='com.google.guava', module='guava', version='26.0-jre')
])

import com.amazonaws.auth.AWSCredentials
import com.amazonaws.auth.DefaultAWSCredentialsProviderChain
import com.amazonaws.auth.profile.ProfileCredentialsProvider
import com.amazonaws.regions.Regions
import com.amazonaws.services.rekognition.AmazonRekognition
import com.amazonaws.services.rekognition.AmazonRekognitionClientBuilder
import com.amazonaws.services.rekognition.model.IndexFacesRequest
import com.amazonaws.services.rekognition.model.CreateCollectionRequest
import com.amazonaws.services.rekognition.model.DeleteCollectionRequest
import com.amazonaws.services.rekognition.model.Face
import com.amazonaws.services.rekognition.model.FaceMatch
import com.amazonaws.services.rekognition.model.Image
import com.amazonaws.services.rekognition.model.ListCollectionsRequest
import com.amazonaws.services.rekognition.model.ListFacesRequest
import com.amazonaws.services.rekognition.model.SearchFacesRequest

import com.google.common.base.Stopwatch
import com.google.common.collect.HashMultimap

import java.io.FilenameFilter
import java.nio.ByteBuffer
import java.util.concurrent.TimeUnit
import java.util.List

import groovy.json.JsonOutput
import groovy.transform.Canonical

import org.apache.tika.Tika

def defaultCollectionId = 'ephemeral-faces-collection'
def defaultMatchConfidenceThreshold = 80F

def printErr = System.err.&println

def cli = new CliBuilder(header: 'AWS Rekognition Match Faces POC', usage:'./matchFaces [options] <directoryOfImagesWithFaces>', width: 140, footer: 'Note: The AWS Rekognition API only supports facial recognition for JPEG and PNG image formats. This script will select on those formats for index operations by Rekognition.')
cli.collectionId(args:1, argName:'id', "The collection id to use for this execution [${defaultCollectionId}]")
cli.delete('Delete the collection of faces post processing')
cli.forceRecreate('Force collection re-creation, if the collection exists')
cli.help('Show this menu')
cli.matchConfidenceThreshold(args:1, argName:'threshold', 'The match confidence threshold to use, values 0-100 [80]')
cli.saveImageData('Save image data as JSON for each response from AWS index image operation')
cli.verbose('Verbose output')

def cliOptions = cli.parse(args)

if (cliOptions.help || cliOptions.arguments().size() != 1) {
  cli.usage()
  System.exit(0)
}

def collectionId = cliOptions?.collectionId ?: defaultCollectionId
def matchConfidenceThreshold = cliOptions?.matchConfidenceThreshold ? Float.parseFloat(cliOptions.matchConfidenceThreshold) : defaultMatchConfidenceThreshold
def deleteCollection = cliOptions.delete
def forceRecreate = cliOptions.forceRecreate
def saveImageData = cliOptions.saveImageData
def verboseOutput = cliOptions.verbose

class ImageFilter implements FilenameFilter {

  def acceptableImageTypes = ['image/jpeg', 'image/png']
  def tika = new Tika()

  public boolean accept(File file, String filename) {
    acceptableImageTypes.contains(tika.detect(file))
  }
}

def sourceImages = new File(cliOptions.arguments().first())

if (matchConfidenceThreshold < 0 || matchConfidenceThreshold > 100) {
  printErr "The supplied matchConfidenceThreshold of ${matchConfidenceThreshold} is invalid. Please supply a value between 0-100 or remove the argument to use the default value of 80."
  System.exit(-1)
}

if (!sourceImages.exists()) {
  printErr "The directory '${sourceImages.path}' does not exist."
  System.exit(-1)
}

def listOfImages = sourceImages.listFiles(new ImageFilter())

if (!listOfImages) {
  printErr "There are no images suitable for processing in the directory '${sourceImages.path}'"
  System.exit(0)
}

def awsRekognitionClient = AmazonRekognitionClientBuilder
   .standard()
   .withRegion(Regions.US_WEST_2)
   .withCredentials(new DefaultAWSCredentialsProviderChain())
   .build()

def collectionExists = awsRekognitionClient.listCollections(new ListCollectionsRequest()).collectionIds.contains(collectionId)

if (collectionExists && forceRecreate) {
  awsRekognitionClient.deleteCollection(new DeleteCollectionRequest().withCollectionId(collectionId))
  awsRekognitionClient.createCollection(new CreateCollectionRequest().withCollectionId(collectionId))
} else if (!collectionExists) {
  awsRekognitionClient.createCollection(new CreateCollectionRequest().withCollectionId(collectionId))
}

def existingFaces = awsRekognitionClient.listFaces(new ListFacesRequest().withCollectionId(collectionId)).faces

def existingFacesExternalImageIds = existingFaces.collect { face ->
  face.externalImageId
}

def totalProcessingStopwatch = Stopwatch.createStarted()
def perImageProcessingStopwatch = Stopwatch.createUnstarted()

listOfImages.each { file ->

  if (existingFacesExternalImageIds.contains(file.name)) {

    if (verboseOutput) {
      println "The image file ${file.name} has already been indexed. Skipping."
    }
  } else {

    perImageProcessingStopwatch.start()

    def indexFacesRequest = new IndexFacesRequest()
       .withImage(new Image()
         .withBytes(ByteBuffer.wrap(file.getBytes())))
       .withCollectionId(collectionId)
       .withExternalImageId(file.name)
       .withDetectionAttributes('ALL')

     def indexFacesResponse = awsRekognitionClient.indexFaces(indexFacesRequest)

     perImageProcessingStopwatch.stop()

     if (verboseOutput) {
       println "Found ${indexFacesResponse.faceRecords.size()} faces in image ${file.name} in ${perImageProcessingStopwatch.elapsed(TimeUnit.MILLISECONDS)} ms"
     } else if (indexFacesResponse.faceRecords.size() != 1) {
       println "Found ${indexFacesResponse.faceRecords.size()} faces in image ${file.name} in ${perImageProcessingStopwatch.elapsed(TimeUnit.MILLISECONDS)} ms"
     }

     if (saveImageData) {
       new File("${sourceImages.path}/${file.name}_results.json").write(JsonOutput.prettyPrint(JsonOutput.toJson(indexFacesResponse)))
     }

     perImageProcessingStopwatch.reset()
  }
}

totalProcessingStopwatch.stop()

if (verboseOutput) {
  println "Took a total of ${totalProcessingStopwatch.elapsed(TimeUnit.SECONDS)} seconds to index ${listOfImages.size()} images"
}

println "/----------------------------------------------------------------------------------------------------------------------------------------\\"
println "| Source Face ID                       | Source File          | Target Face ID                       | Target File          | Similarity |"
println "|----------------------------------------------------------------------------------------------------------------------------------------|"

def leftAlignedFormatOutput = "| %s | %-20s | %s | %-20s | %-10f |%n"

existingFaces.each { face ->

    awsRekognitionClient
      .searchFaces(new SearchFacesRequest()
        .withCollectionId(collectionId)
        .withFaceId(face.faceId)
        .withFaceMatchThreshold(matchConfidenceThreshold))
      .faceMatches.each { faceMatch ->

        System.out.format(leftAlignedFormatOutput, face.faceId, face.externalImageId, faceMatch.face.faceId, faceMatch.face.externalImageId, faceMatch.similarity)
      }
  }

println "\\----------------------------------------------------------------------------------------------------------------------------------------/"

if (deleteCollection) {
  awsRekognitionClient.deleteCollection(new DeleteCollectionRequest().withCollectionId(collectionId))
}
