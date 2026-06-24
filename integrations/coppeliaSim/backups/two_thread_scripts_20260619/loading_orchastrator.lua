sim = require('sim')

local apiScript = -1
local placedItems = {}
local lastFull = false
local commandTimeoutSteps = 12000

local function obj(alias)
    local handle = sim.getObject(alias, {noError = true})
    return handle and handle >= 0 and handle or -1
end

local function findExternalApi()
    local aliases = {
        sim.getStringSignal('externalApiAlias') or '',
        '/external_api',
        ':/external_api',
        '/mobile_arm/external_api',
        ':/mobile_arm/external_api',
    }
    local types = {sim.scripttype_childscript, sim.scripttype_customizationscript}

    for _, alias in ipairs(aliases) do
        if alias ~= '' then
            local objectHandle = obj(alias)
            if objectHandle >= 0 then
                for _, scriptType in ipairs(types) do
                    local scriptHandle = sim.getScript(scriptType, objectHandle)
                    if scriptHandle and scriptHandle >= 0 then
                        return scriptHandle
                    end
                end
            end
        end
    end

    print('external_api missing: attach external_api.lua to /external_api or set externalApiAlias')
    return -1
end

local function api(name, data)
    if apiScript < 0 then
        return {ok = false, reason = 'external_api missing'}
    end

    return sim.callScriptFunction(name, apiScript, data or {})
end

local function itemHandle(item)
    return type(item) == 'table' and item.handle or item
end

local function itemStillExists(item)
    local handle = itemHandle(item)
    if not handle then
        return false
    end
    return sim.getObjectPosition(handle, sim.handle_world) ~= nil
end

local function itemsIfFull()
    local packed = sim.getStringSignal('trackedItemsData')
    if not packed then
        return nil
    end

    local data = sim.unpackTable(packed)
    if not data or data.full ~= true or type(data.items) ~= 'table' then
        return nil
    end

    local items = {}
    for _, item in ipairs(data.items) do
        if item.handle and itemStillExists(item) then
            items[#items + 1] = item
        end
    end

    return #items > 0 and items or nil
end

local function isCuboid(handle)
    local alias = sim.getObjectAlias(handle, 1)
    local localAlias = alias and (alias:match('[^/]+$') or alias)
    return localAlias and localAlias:match('^Cuboid') ~= nil
end

local function trackedCuboidItems(trackedItems)
    local items = {}
    for _, item in ipairs(trackedItems) do
        local handle = itemHandle(item)
        if handle and isCuboid(handle) and not placedItems[handle] then
            items[#items + 1] = item
        end
    end
    return #items > 0 and items or nil
end

local function setRobotPicking(enabled)
    sim.setInt32Signal('robotPicking', enabled and 1 or 0)
end

local function waitExternalApiStatus(field, doneStatus, failedStatus, timeoutSteps)
    for _ = 1, timeoutSteps do
        local state = api('getState')
        if state.ok == false then
            return false, state.reason
        end

        local status = state[field]
        if status == doneStatus then
            return true, status, state
        end
        if status == failedStatus or status == 'stopped' then
            return false, status, state
        end

        sim.step()
    end

    return false, 'timeout'
end

local function statusFailureReason(prefix, status, state, resultField)
    local result = state and state[resultField]
    if result and result.reason then
        return prefix .. ' failed: ' .. tostring(result.reason)
    end
    return prefix .. ' ' .. tostring(status)
end

local function pickAndPlace(item)
    setRobotPicking(true)

    local pickResult = api('pick', {object = itemHandle(item)})
    if not pickResult or not pickResult.ok then
        return false, pickResult and pickResult.reason or 'pick call failed'
    end

    local pickDone, pickStatus, pickState = waitExternalApiStatus('pickStatus', 'done', 'failed', commandTimeoutSteps)
    if not pickDone then
        return false, statusFailureReason('pick', pickStatus, pickState, 'lastPickResult')
    end

    local placeResult = api('place', {position = {0, 0, 1}})
    if not placeResult or not placeResult.ok then
        return false, placeResult and placeResult.reason or 'place call failed'
    end

    local placeDone, placeStatus, placeState = waitExternalApiStatus('placeStatus', 'done', 'failed', commandTimeoutSteps)
    if not placeDone then
        return false, statusFailureReason('place', placeStatus, placeState, 'lastPlaceResult')
    end

    placedItems[itemHandle(item)] = true
    return true
end

function sysCall_init()
    math.randomseed(os.time())
    apiScript = findExternalApi()
    commandTimeoutSteps = sim.getInt32Signal('externalApiCommandTimeoutSteps') or commandTimeoutSteps
    placedItems = {}
    lastFull = false
    setRobotPicking(false)
    print('loading_test ready, externalApiScript=', apiScript)
end

function sysCall_thread()
    while sim.getSimulationState() ~= sim.simulation_advancing_abouttostop do
        local fullItems = itemsIfFull()
        local items = fullItems and trackedCuboidItems(fullItems)

        if items and not lastFull then
            local item = items[math.random(#items)]
            local ok, reason = pickAndPlace(item)
            print('loading_test item=', itemHandle(item), 'ok=', ok, 'reason=', reason or '')
            setRobotPicking(false)
        end

        lastFull = fullItems ~= nil
        sim.step()
    end
end

function sysCall_cleanup()
    setRobotPicking(false)
end
