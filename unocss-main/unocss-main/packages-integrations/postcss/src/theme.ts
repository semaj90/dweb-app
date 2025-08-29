import type { UnoGenerator } from '@unocss/core'
import type { Root } from 'postcss'
import { transformThemeFn } from '@unocss/rule-utils'

export async function parseTheme(root: Root, uno: UnoGenerator): Promise<any> {
  root.walkDecls((decl) => {
    decl.value = transformThemeFn(decl.value, uno.config.theme)
  })
}
