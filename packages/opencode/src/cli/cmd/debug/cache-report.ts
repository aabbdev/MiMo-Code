import type { Argv } from "yargs"
import { Database } from "../../../storage"
import { Database as BunDatabase } from "bun:sqlite"
import { cmd } from "../cmd"
import {
  DEFAULT_CACHE_TTL_MS,
  cacheReport,
  renderCacheReport,
  type CacheReportRow,
} from "../../../session/cache-report"

/**
 * Phase-0 instrument: decompose a window of real calls into the buckets that name
 * WHY their tokens were charged at the uncached rate. The analysis is pure
 * (session/cache-report.ts); this command only reads the trajectory DB readonly and
 * prints. Without it, every cost change shipped into the harness is judged on a
 * hand-built query — which is exactly how the cold-resume finding was found, and
 * exactly what should not have to be redone by hand to verify it.
 */
export const CacheReportCommand = cmd({
  command: "cache-report [session]",
  describe: "decompose uncached token spend by cause (new content / cold resume / profile or model switch)",
  builder: (yargs: Argv) =>
    yargs
      .positional("session", {
        type: "string",
        describe: "restrict to one session id",
      })
      .option("days", {
        type: "number",
        default: 3,
        describe: "window in days back from now",
      })
      .option("ttl", {
        type: "number",
        default: DEFAULT_CACHE_TTL_MS,
        describe: "provider cache TTL in ms; idle beyond it counts as a cold resume",
      })
      .option("limit", {
        type: "number",
        default: 10,
        describe: "largest uncached calls to list",
      })
      .option("json", {
        type: "boolean",
        default: false,
        describe: "emit the raw report instead of the text render",
      }),
  handler: async (args: {
    session?: string
    days: number
    ttl: number
    limit: number
    json: boolean
  }) => {
    const since = Date.now() - args.days * 86_400_000
    const db = new BunDatabase(Database.Path, { readonly: true })
    try {
      const rows = db
        .query(
          `SELECT m.session_id AS sid, m.time_created AS t,
                  json_extract(m.data,'$.agent') AS agent,
                  json_extract(m.data,'$.model.providerID') AS prov,
                  json_extract(m.data,'$.model.modelID') AS mid,
                  json_extract(p.data,'$.tokens.input') AS unc,
                  json_extract(p.data,'$.tokens.cache.read') AS cr,
                  json_extract(p.data,'$.tokens.cache.write') AS cw,
                  json_extract(p.data,'$.cost') AS cost
           FROM part p JOIN message m ON p.message_id = m.id
           WHERE json_extract(p.data,'$.type')='step-finish' AND m.time_created >= ?
           ORDER BY m.time_created`,
        )
        .all(since) as Array<{
        sid: string
        t: number
        agent: string | null
        prov: string | null
        mid: string | null
        unc: number | null
        cr: number | null
        cw: number | null
        cost: number | null
      }>

      const report = cacheReport({
        rows: rows
          .filter((r) => !args.session || r.sid === args.session)
          .map((r) => ({
            sessionID: r.sid,
            time: r.t,
            agent: r.agent ?? "-",
            model: `${r.prov ?? "-"}/${r.mid ?? "-"}`,
            uncached: r.unc ?? 0,
            cached: r.cr ?? 0,
            written: r.cw ?? 0,
            cost: r.cost ?? 0,
          })),
        ttl: args.ttl,
      })

      if (args.json) console.log(JSON.stringify(report, null, 2))
      else console.log(renderCacheReport(report, { limit: args.limit }))
    } finally {
      db.close()
    }
  },
})

export type { CacheReportRow }