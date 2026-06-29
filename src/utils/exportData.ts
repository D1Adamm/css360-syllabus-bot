import type { EvaluationRecord, SeedExample } from '../types';

function downloadFile(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export function exportAsJson<T>(data: T[], filename: string): void {
  const content = JSON.stringify(data, null, 2);
  downloadFile(content, filename, 'application/json');
}

export function exportAsJsonl(seeds: SeedExample[], filename: string): void {
  const content = seeds.map((seed) => JSON.stringify(seed)).join('\n');
  downloadFile(content, filename, 'application/x-ndjson');
}

export function exportFilteredJson(seeds: SeedExample[]): void {
  exportAsJson(seeds, 'syllabus-seed-data-filtered.json');
}

export function exportFilteredJsonl(seeds: SeedExample[]): void {
  exportAsJsonl(seeds, 'syllabus-seed-data-filtered.jsonl');
}

export function exportCompleteJsonl(seeds: SeedExample[]): void {
  exportAsJsonl(seeds, 'syllabus-seed-data.jsonl');
}

export function exportUserSeedsJsonl(seeds: SeedExample[]): void {
  exportAsJsonl(seeds, 'syllabus-user-seed-data.jsonl');
}

export function exportEvaluationsJson(evaluations: EvaluationRecord[]): void {
  exportAsJson(evaluations, 'syllabus-evaluations.json');
}
