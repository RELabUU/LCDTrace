def calculateTimeDifference(startTime, endTime):
    #Find out the time difference in seconds
    timeDifferenceInSeconds = (endTime - startTime)
    
    #Translate the difference in minutes and seconds
    minutes = round(timeDifferenceInSeconds / 60)
    seconds = timeDifferenceInSeconds % 60
    
    #Create string to print
    stringToPrint = (str(minutes) + " minutes and " + str(seconds) + " seconds")
    return(stringToPrint)