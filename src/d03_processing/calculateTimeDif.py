#Calculate the time difference between 2 dates in seconds
def calculateTimeDif(datetimeA, datetimeB):   
    # Get the difference between datetimes (as timedelta)
    dateTimeDelta = datetimeA - datetimeB

    # Find Delta in seconds
    dateTimeDeltaInSeconds = dateTimeDelta.total_seconds()
    
    return(dateTimeDeltaInSeconds)