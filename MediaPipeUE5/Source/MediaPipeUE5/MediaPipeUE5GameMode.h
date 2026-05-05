// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "MediaPipeUE5GameMode.generated.h"

UCLASS(minimalapi)
class AMediaPipeUE5GameMode : public AGameModeBase
{
	GENERATED_BODY()

public:
	AMediaPipeUE5GameMode();

	virtual void StartPlay() override;

	UFUNCTION(BlueprintImplementableEvent, BlueprintCallable)
	void PlayMediaPipe();

	UPROPERTY(BlueprintReadOnly, EditAnywhere)
	float FrameInterval = 0.05;

	UFUNCTION(BlueprintCallable)
	void GoToNextFrame();

	UPROPERTY(BlueprintReadOnly)
	int32 FrameNumber = 0;

	int32 TotalFrames = 0;

	UPROPERTY(EditAnywhere)
	FString MediaPipeDirPath = "C:\\Users\\zheng\\Documents\\BachelorThesisMediaPipe\\MediaPipePython\\";
};



