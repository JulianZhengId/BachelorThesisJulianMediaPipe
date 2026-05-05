// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Pawn.h"
#include "MediaPipePawn.generated.h"

UCLASS()
class MEDIAPIPEUE5_API AMediaPipePawn : public APawn
{
	GENERATED_BODY()

public:
	AMediaPipePawn();

protected:
	virtual void BeginPlay() override;
};
