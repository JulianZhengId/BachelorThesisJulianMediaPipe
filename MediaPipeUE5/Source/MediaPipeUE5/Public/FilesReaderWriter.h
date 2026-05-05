#pragma once

class FilesReaderWriter
{
public:
	static bool ReadFile(const FString& Filepath, TArray<FString>& ResultStrings);
};
