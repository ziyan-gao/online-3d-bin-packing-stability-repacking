sim = require('sim')

local apiScript = -1
local placedItems = {}
local lastFull = false
local commandTimeoutSteps = 12000
local activeItem = nil
local activePhase = 'idle'
local activeCommandSteps = 0

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

local function statusFailureReason(prefix, status, state, resultField)
    local result = state and state[resultField]
    if result and result.reason then
        return prefix .. ' failed: ' .. tostring(result.reason)
    end
    return prefix .. ' ' .. tostring(status)
end

local function resetActiveCommand()
    activeItem = nil
    activePhase = 'idle'
    activeCommandSteps = 0
    setRobotPicking(false)
end

local function finishActiveCommand(ok, reason)
    local item = activeItem
    if item then
        if ok then
            placedItems[itemHandle(item)] = true
        end
        print('loading_test item=', itemHandle(item), 'ok=', ok, 'reason=', reason or '')
    end
    resetActiveCommand()
end

local function beginPickAndPlace(item)
    activeItem = item
    activePhase = 'wait_pick'
    activeCommandSteps = 0
    setRobotPicking(true)

    local pickResult = api('pick', {object = itemHandle(item)})
    if not pickResult or not pickResult.ok then
        finishActiveCommand(false, pickResult and pickResult.reason or 'pick call failed')
        return
    end
end

local function pollActiveCommand()
    if activePhase == 'idle' then
        return
    end

    activeCommandSteps = activeCommandSteps + 1
    if activeCommandSteps > commandTimeoutSteps then
        finishActiveCommand(false, activePhase .. ' timeout')
        return
    end

    local state = api('getState')
    if state.ok == false then
        finishActiveCommand(false, state.reason)
        return
    end

    if activePhase == 'wait_pick' then
        local pickStatus = state.pickStatus
        if pickStatus == 'done' then
            local placeResult = api('place', {position = {0, 0, 1}})
            if not placeResult or not placeResult.ok then
                finishActiveCommand(false, placeResult and placeResult.reason or 'place call failed')
                return
            end
            activePhase = 'wait_place'
            activeCommandSteps = 0
            return
        end
        if pickStatus == 'failed' or pickStatus == 'stopped' then
            finishActiveCommand(false, statusFailureReason('pick', pickStatus, state, 'lastPickResult'))
            return
        end
    elseif activePhase == 'wait_place' then
        local placeStatus = state.placeStatus
        if placeStatus == 'done' then
            finishActiveCommand(true)
            return
        end
        if placeStatus == 'failed' or placeStatus == 'stopped' then
            finishActiveCommand(false, statusFailureReason('place', placeStatus, state, 'lastPlaceResult'))
            return
        end
    end
end

local function maybeStartPickAndPlace()
    local fullItems = itemsIfFull()
    local items = fullItems and trackedCuboidItems(fullItems)

    if items and not lastFull then
        local item = items[math.random(#items)]
        beginPickAndPlace(item)
    end

    lastFull = fullItems ~= nil
end

function sysCall_init()
    math.randomseed(os.time())
    apiScript = findExternalApi()
    commandTimeoutSteps = sim.getInt32Signal('externalApiCommandTimeoutSteps') or commandTimeoutSteps
    placedItems = {}
    lastFull = false
    activeItem = nil
    activePhase = 'idle'
    activeCommandSteps = 0
    setRobotPicking(false)
    print('loading_test ready, externalApiScript=', apiScript)
end

function sysCall_actuation()
    pollActiveCommand()
    if activePhase == 'idle' then
        maybeStartPickAndPlace()
    end
end

function sysCall_cleanup()
    resetActiveCommand()
end
