export interface FilterState {
  date_ranges: Record<string, [string,string]>
  year_ranges: Record<string, [number,number]>
  category_picks: Record<string, string[]>
  numeric_ranges: Record<string, [number,number]>
  search_col?: string
  search_text?: string
}

export type Role = 'date'|'year'|'numeric'|'binary'|'boolean'|'category'|'text'|'id'
