#include "MediaPipeUE5/Public/FilesReaderWriter.h"

bool FilesReaderWriter::ReadFile(const FString& Filepath, TArray<FString>& ResultStrings)
{
	//GEngine->AddOnScreenDebugMessage(-1, 2.0f, FColor::Yellow,FString::Printf(TEXT("Reading file: %s"), *Filepath));

	//check file exists
	if (!FPlatformFileManager::Get().GetPlatformFile().FileExists(*Filepath))
	{
		GEngine->AddOnScreenDebugMessage(-1, 2.0f, FColor::Yellow,FString::Printf(TEXT("File does not exist on path: %s"), *Filepath));
		return false;
	}

	//try to read file
	if (!FFileHelper::LoadFileToStringArray(ResultStrings, *Filepath))
	{
		GEngine->AddOnScreenDebugMessage(-1, 2.0f, FColor::Yellow,FString::Printf(TEXT("Failed to load file tos string array: %s"), *Filepath));
		return false;
	}

	return true;
}
