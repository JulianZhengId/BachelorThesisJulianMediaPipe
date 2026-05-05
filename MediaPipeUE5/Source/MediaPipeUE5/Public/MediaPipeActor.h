// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "MediaPipeActor.generated.h"

UCLASS()
class MEDIAPIPEUE5_API AMediaPipeActor : public AActor
{
	GENERATED_BODY()
	
public:	
	AMediaPipeActor();

protected:
	virtual void BeginPlay() override;
};
