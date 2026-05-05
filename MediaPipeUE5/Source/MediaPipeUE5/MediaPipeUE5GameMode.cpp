// Copyright Epic Games, Inc. All Rights Reserved.

#include "MediaPipeUE5GameMode.h"
#include "MediaPipeUE5Character.h"
#include "UObject/ConstructorHelpers.h"

AMediaPipeUE5GameMode::AMediaPipeUE5GameMode()
{

}

void AMediaPipeUE5GameMode::StartPlay()
{
	Super::StartPlay();

	FrameNumber = 0;
	PlayMediaPipe();
}

void AMediaPipeUE5GameMode::GoToNextFrame()
{
	FrameNumber = (FrameNumber + 1) % TotalFrames;
}
