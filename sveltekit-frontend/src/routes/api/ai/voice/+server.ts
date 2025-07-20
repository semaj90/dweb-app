import { SpeechService } from "$lib/services/speech-service";
import { json } from "@sveltejs/kit";

export async function POST({ request }) {
  const { audio } = await request.json();
  const transcript = await SpeechService.transcribe(audio);
  return json({ transcript });
}
