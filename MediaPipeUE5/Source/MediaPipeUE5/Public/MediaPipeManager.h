#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "MediaPipeManager.generated.h"

class AMediaPipeUE5GameMode;

USTRUCT(BlueprintType)
struct FLandmarksDataPerFrame
{
	GENERATED_BODY()

	UPROPERTY()
	TArray<FVector> Landmarks;
};

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class MEDIAPIPEUE5_API UMediaPipeManager : public UActorComponent
{
	GENERATED_BODY()

public:
	UMediaPipeManager();
	
	UFUNCTION(BlueprintCallable, BlueprintPure)
	const TArray<FVector>& GetLandmarksData();

	UFUNCTION(BlueprintCallable)
	FQuat GetEulerZYXOrientation() const;
	
	UPROPERTY(EditAnywhere)
	bool NegateX = false;

	UPROPERTY(EditAnywhere)
	bool NegateY = false;

	UPROPERTY(EditAnywhere)
	bool NegateZ = false;
	
	UPROPERTY(BlueprintReadOnly)
	TArray<FLandmarksDataPerFrame> LandmarksData;

	UPROPERTY(EditInstanceOnly)
	FString FileName = "";

	UPROPERTY(BlueprintReadWrite)
	AMediaPipeUE5GameMode* MediaPipeGameMode = nullptr;

protected:
	virtual void BeginPlay() override;

private:
	void InitialSetup();

	bool ReadMediaPipeLandmarks();

	void RefineMediaPipeLandmarks(TArray<FVector>& RawLandmarks, TArray<FVector>& RefinedLandmarks) const;

	UPROPERTY()
	USkeletalMeshComponent* Human = nullptr;

	float HumanReferenceDistanceHorizontal = 0;
	float HumanReferenceDistanceVertical = 0;

	FVector NegatingVector;
};
