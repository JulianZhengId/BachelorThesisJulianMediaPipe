#include "MediaPipeUE5/Public/MediaPipeManager.h"

#include "Kismet/GameplayStatics.h"
#include "MediaPipeUE5/MediaPipeUE5GameMode.h"
#include "MediaPipeUE5/Public/FilesReaderWriter.h"


UMediaPipeManager::UMediaPipeManager()
{
	PrimaryComponentTick.bCanEverTick = false;
}

const TArray<FVector>& UMediaPipeManager::GetLandmarksData()
{
	return LandmarksData[MediaPipeGameMode->FrameNumber].Landmarks;
}

FQuat UMediaPipeManager::GetEulerZYXOrientation() const
{
	TArray<FVector> Landmarks = LandmarksData[MediaPipeGameMode->FrameNumber].Landmarks;
	const FVector Wrist = Landmarks[0];
	const FVector IndexMCP = Landmarks[5];
	const FVector MiddleMCP = Landmarks[9];
	const FVector PinkyMCP = Landmarks[17];
	
	const FVector PalmForward = (MiddleMCP - Wrist).GetSafeNormal();
	const FVector PalmSide = (IndexMCP - PinkyMCP).GetSafeNormal();
	
	const FMatrix RotMatrix = FRotationMatrix::MakeFromXZ(-PalmForward, PalmSide);
	
	const FQuat WristQuat = FQuat(RotMatrix);
	const FQuat Corrector = WristQuat;
	return FQuat(Corrector.X, Corrector.Y, Corrector.Z, Corrector.W).GetNormalized();
}

void UMediaPipeManager::BeginPlay()
{
	Super::BeginPlay();
	InitialSetup();
	ReadMediaPipeLandmarks();
}

void UMediaPipeManager::InitialSetup()
{
	MediaPipeGameMode = Cast<AMediaPipeUE5GameMode>(UGameplayStatics::GetGameMode(GetWorld()));
	if (!MediaPipeGameMode)
	{
		UE_LOG(LogTemp, Error, TEXT("MediaPipeGameMode is null"));
		return;
	}
	
	Human = GetOwner()->FindComponentByClass<USkeletalMeshComponent>();
	if (Human)
	{
		const FVector HumanWrist = Human->GetBoneLocation(FName("hand_r"));
		const FVector HumanMiddleMCP = Human->GetBoneLocation(FName("middle_01_r"));
		HumanReferenceDistanceVertical = FVector::Distance(HumanWrist, HumanMiddleMCP);

		const FVector HumanIndexMCP = Human->GetBoneLocation(FName("index_01_r"));
		const FVector HumanPinkyMCP = Human->GetBoneLocation(FName("pinky_01_r"));
		HumanReferenceDistanceHorizontal = FVector::Distance(HumanIndexMCP, HumanPinkyMCP);
	}

	NegatingVector = FVector(NegateX ? -1 : 1, NegateY ? -1 : 1, NegateZ ? -1 : 1);
	LandmarksData.Empty();
}

bool UMediaPipeManager::ReadMediaPipeLandmarks()
{
	TArray<FString> DataLines;
	if (!FilesReaderWriter::ReadFile(FPaths::Combine(MediaPipeGameMode->MediaPipeDirPath, FileName), DataLines))
	{
		UE_LOG(LogTemp, Error, TEXT("%s"), *FileName);
		return false;
	}

	//read each lines
	for (int i = 1; i < DataLines.Num(); i++)
	{
		//set frame number, might useful later
		FLandmarksDataPerFrame LandmarksDataPerFrame;

		//split line by comma
		TArray<FString> LandmarksPerFrameAsStrings;
		DataLines[i].ParseIntoArray(LandmarksPerFrameAsStrings, TEXT(","));

		//get vectors from the line
		TArray<FVector> RawLandmarks;
		for (int j = 0; j < LandmarksPerFrameAsStrings.Num(); j += 3)
		{
			RawLandmarks.Add(FVector(FCString::Atof(*LandmarksPerFrameAsStrings[j]), FCString::Atof(*LandmarksPerFrameAsStrings[j + 1]), FCString::Atof(*LandmarksPerFrameAsStrings[j + 2])));
		}

		//refine landmarks by scaling and subtracting from wrist transform
		RefineMediaPipeLandmarks(RawLandmarks, LandmarksDataPerFrame.Landmarks);

		//store to array
		LandmarksData.Add(LandmarksDataPerFrame);
	}

	MediaPipeGameMode->TotalFrames = DataLines.Num() - 1;
	return true;
}

void UMediaPipeManager::RefineMediaPipeLandmarks(TArray<FVector>& RawLandmarks, TArray<FVector>& RefinedLandmarks) const
{
	//do scaling
	const FVector MediaPipeWrist = RawLandmarks[0];
	const FVector MediaPipeMiddleMCP = RawLandmarks[9];
	const float MediaPipeReferenceDistanceVertical = FVector::Distance(MediaPipeWrist, MediaPipeMiddleMCP);
	const float VerticalScalingFactor = HumanReferenceDistanceVertical / MediaPipeReferenceDistanceVertical;

	const FVector MediaPipeIndexMCP = RawLandmarks[5];
	const FVector MediaPipePinkyMCP = RawLandmarks[17];
	const float MediaPipeReferenceDistanceHorizontal = FVector::Distance(MediaPipeIndexMCP, MediaPipePinkyMCP);
	const float HorizontalScalingFactor = HumanReferenceDistanceHorizontal / MediaPipeReferenceDistanceHorizontal;
	
	const FVector InitialPos = Human->GetBoneLocation("hand_r");
	RefinedLandmarks.Add(InitialPos);

	for (int i = 1; i < RawLandmarks.Num(); i++)
	{
		RefinedLandmarks.Add(InitialPos + RawLandmarks[i] * NegatingVector * FVector(VerticalScalingFactor, (HorizontalScalingFactor + VerticalScalingFactor) / 2.f, HorizontalScalingFactor));
	}
}
