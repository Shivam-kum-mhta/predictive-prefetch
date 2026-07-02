export async function loadNews() {
  // Fetch the news.txt file from the public directory
  const response = await fetch('/news.txt');
  const text = await response.text();

  // Find the first '{' to skip any non-JSON lines (like the GET ... line)
  const jsonStart = text.indexOf('{');
  if (jsonStart === -1) throw new Error('No JSON found in news.txt');

  // Parse the JSON part
  const jsonString = text.slice(jsonStart);
  const data = JSON.parse(jsonString);

  // Return the articles array
  return data.predictions || [];
}