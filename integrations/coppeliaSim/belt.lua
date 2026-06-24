function sysCall_init()
    sim = require('sim')
    simUI = require('simUI')

    initConfig()
    initHandles()
    initBoxes()
    initItemTrackingUi()

    rebuildBuffer(segmentCount)

    sim.setStepping(true)
    currentTime = sim.getSimulationTime()
end

function initConfig()
    segmentCount = sim.getInt32Signal('segmentCount') or 5
    conveyorVelocity = sim.getFloatSignal('conveyorVelocity') or 0.3
    dropInterval = sim.getFloatSignal('dropInterval') or 4.0
    xPad = sim.getFloatSignal('xPad') or 0.4
    yPad = sim.getFloatSignal('yPad') or 0.4
    zMin = sim.getFloatSignal('zMin') or 0.15
    zMax = sim.getFloatSignal('zMax') or 1.20

    lastSegmentCount = segmentCount

    segmentPitch = xPad
    spawnEnabled = true
    spawnPos = {-1.25, 0.375, 0.7}

    sim.setFloatSignal('defaultVel', effectiveConveyorVelocity())
end

function syncConfigFromSignals()
    local newSegmentCount = sim.getInt32Signal('segmentCount') or segmentCount
    local newConveyorVelocity = sim.getFloatSignal('conveyorVelocity') or conveyorVelocity
    local newDropInterval = sim.getFloatSignal('dropInterval') or dropInterval
    xPad = sim.getFloatSignal('xPad') or xPad
    yPad = sim.getFloatSignal('yPad') or yPad
    zMin = sim.getFloatSignal('zMin') or zMin
    zMax = sim.getFloatSignal('zMax') or zMax
    segmentCount = math.floor(newSegmentCount)
    conveyorVelocity = newConveyorVelocity
    dropInterval = newDropInterval

    sim.setFloatSignal('defaultVel', effectiveConveyorVelocity())

    if segmentCount ~= lastSegmentCount then
        rebuildBuffer(segmentCount)
        lastSegmentCount = segmentCount
    end
end

function isRobotPicking()
    local v = sim.getInt32Signal('robotPicking')

    return v ~= nil and v ~= 0
end

function effectiveConveyorVelocity()
    if isRobotPicking() then
        return 0
    end

    return conveyorVelocity
end

function initHandles()
    rootHandle = sim.getObject('.')

    startPos = sim.getObjectPosition(rootHandle)
    startPos[3] = 0.3

    template = sim.getObjectHandle(':/efficientConveyorTemplate')
    templateHandle = sim.getObjectsInTree(template)

    final = sim.getObjectHandle(':/finalConveyorTemplate')
    finalTemplateHandle = sim.getObjectsInTree(final)

    generatedRoot = sim.getObject(':/generatedSegments')
    spawnSensor = sim.getObject(':/spawnSensor')
end

function initBoxes()
    boxTypes = {
        {size = {0.2, 0.18, 0.134}, probability = 0.606, color = randomColor()},
        {size = {0.25, 0.18, 0.14}, probability = 0.095, color = randomColor()},
        {size = {0.25, 0.22, 0.158}, probability = 0.126, color = randomColor()},
        {size = {0.32, 0.24, 0.210}, probability = 0.135, color = randomColor()},
        {size = {0.32, 0.28, 0.257}, probability = 0.038, color = randomColor()},
    }
end

function randomColor()
    return {math.random(), math.random(), math.random()}
end

function rebuild(inData)
    local n = inData.count or segmentCount

    sim.setInt32Signal('segmentCount', n)
    segmentCount = n

    rebuildBuffer(segmentCount)
    lastSegmentCount = segmentCount

    return {
        ok = true,
        segmentCount = segmentCount,
    }
end

function rebuildBuffer(n)
    clearGeneratedSegments()

    segmentHandles = {}

    for i = 0, n - 2 do
        local h = sim.copyPasteObjects(templateHandle, 0)[1]
        sim.setObjectParent(h, generatedRoot, true)

        local x = startPos[1] + i * segmentPitch
        local y = startPos[2]
        local z = startPos[3]

        sim.setObjectPosition(h, {x, y, z}, sim.handle_world)
        setSegmentColorOddEven(h, i)
        sim.setObjectAlias(h, 'efficientConveyor[' .. i .. ']', 0)

        table.insert(segmentHandles, h)
    end

    local finalIndex = n - 1
    local h = sim.copyPasteObjects(finalTemplateHandle, 0)[1]
    sim.setObjectParent(h, generatedRoot, true)

    local x = startPos[1] + finalIndex * segmentPitch
    local y = startPos[2]
    local z = startPos[3]

    sim.setObjectPosition(h, {x, y, z}, sim.handle_world)
    setSegmentColorOddEven(h, finalIndex)
    sim.setObjectAlias(h, 'efficientConveyor[' .. finalIndex .. ']', 0)

    table.insert(segmentHandles, h)

    segmentCount = n
end

function clearGeneratedSegments()
    local children = sim.getObjectsInTree(generatedRoot, sim.handle_all, 1)

    for i = 1, #children do
        sim.removeObject(children[i])
    end
end

function getChildByAlias(root, alias)
    local objs = sim.getObjectsInTree(root, sim.handle_all, 0)

    for i = 1, #objs do
        local objAlias = sim.getObjectAlias(objs[i], 1)

        if objAlias == alias then
            return objs[i]
        end
    end

    return -1
end

function setSegmentColorOddEven(segmentHandle, i)
    local visible1 = getChildByAlias(
        segmentHandle,
        '/generatedSegments/efficientConveyorTemplate/visible1'
    )

    if visible1 == -1 then
        print('visible1 not found for segment ' .. i)
        return
    end

    local color

    if i % 2 == 0 then
        color = {0.85, 0.25, 0.20}
    else
        color = {0.10, 0.65, 0.85}
    end

    sim.setShapeColor(
        visible1,
        nil,
        sim.colorcomponent_ambient_diffuse,
        color
    )
end

function isSpawnBlocked()
    local res, dist, detectP, detectedObj, normVec =
        sim.readProximitySensor(spawnSensor)

    return detectedObj ~= -1
end

function trySpawn()
    if isRobotPicking() then
        sim.setInt32Signal('stopSpawn', 1)
        return false
    end

    if not spawnEnabled then
        sim.setInt32Signal('stopSpawn', 1)
        return false
    end

    if isSpawnBlocked() then
        sim.setInt32Signal('stopSpawn', 1)
        return false
    end

    sim.setInt32Signal('stopSpawn', 0)
    createRandomBox()

    return true
end

function sampleBox()
    local p = math.random()
    local cumulative = 0

    for i = 1, #boxTypes do
        cumulative = cumulative + boxTypes[i].probability

        if p <= cumulative then
            return boxTypes[i]
        end
    end

    return boxTypes[#boxTypes]
end

function createRandomBox()
    local boxSpec = sampleBox()
    local size = boxSpec.size

    local box = sim.createPrimitiveShape(
        sim.primitiveshape_cuboid,
        {size[1], size[2], size[3]},
        2
    )

    sim.setBoolProperty(box, 'dynamic', true)
    sim.setBoolProperty(box, 'respondable', true)
    sim.setBoolProperty(box, 'detectable', true)
    
    sim.setShapeMass(box, 0.1)
    sim.setObjectPosition(box, spawnPos, sim.handle_world)

    sim.setShapeColor(
        box,
        nil,
        sim.colorcomponent_ambient_diffuse,
        boxSpec.color
    )

    sim.resetDynamicObject(box)
    registerTrackedItem(box, size)
end

function getConfig(inData)
    return {
        segmentCount = segmentCount,
        conveyorVelocity = conveyorVelocity,
        dropInterval = dropInterval,
        spawnEnabled = spawnEnabled,
    }
end

function setConfig(inData)
    if inData.segmentCount ~= nil then
        segmentCount = math.floor(inData.segmentCount)
        sim.setInt32Signal('segmentCount', segmentCount)
    end

    if inData.conveyorVelocity ~= nil then
        conveyorVelocity = inData.conveyorVelocity
        sim.setFloatSignal('conveyorVelocity', conveyorVelocity)
        sim.setFloatSignal('defaultVel', effectiveConveyorVelocity())
    end

    if inData.defaultVel ~= nil then
        conveyorVelocity = inData.defaultVel
        sim.setFloatSignal('conveyorVelocity', conveyorVelocity)
        sim.setFloatSignal('defaultVel', effectiveConveyorVelocity())
    end

    if inData.dropInterval ~= nil then
        dropInterval = inData.dropInterval
        sim.setFloatSignal('dropInterval', dropInterval)
    end

    if inData.spawnEnabled ~= nil then
        spawnEnabled = inData.spawnEnabled
    end

    return getConfig({})
end

function spawnOne(inData)
    if isRobotPicking() then
        return {
            ok = false,
            reason = 'robot picking',
        }
    end

    if not spawnEnabled then
        return {
            ok = false,
            reason = 'spawn disabled',
        }
    end

    if isSpawnBlocked() then
        return {
            ok = false,
            reason = 'spawn blocked',
        }
    end

    createRandomBox()

    return {
        ok = true,
    }
end

function getObjectVelocitySafe(handle)
    local linVel, angVel = sim.getObjectVelocity(handle)
    return linVel, angVel
end

function vectorNorm(v)
    return math.sqrt(v[1] * v[1] + v[2] * v[2] + v[3] * v[3])
end

function publishTrackedItemsData()
    local items = {}
    local allItemsStatic = true

    local staticStepThreshold = 20
    local linearStaticThreshold = 0.01
    local angularStaticThreshold = 0.01

    pruneTrackedItems()

    for i = 1, #trackedItemOrder do
        local handle = trackedItemOrder[i]
        local item = trackedItems[handle]
        local pos, quat = objectPose(handle)
        local linVel, angVel = getObjectVelocitySafe(handle)

        if item ~= nil and pos ~= nil and quat ~= nil and linVel ~= nil and angVel ~= nil then
            local linearSpeed = vectorNorm(linVel)
            local angularSpeed = vectorNorm(angVel)

            if item.staticSteps == nil then
                item.staticSteps = 0
            end

            local isCurrentlyStatic =
                linearSpeed < linearStaticThreshold and
                angularSpeed < angularStaticThreshold

            if isCurrentlyStatic then
                item.staticSteps = item.staticSteps + 1
            else
                item.staticSteps = 0
            end

            local itemIsStatic = item.staticSteps >= staticStepThreshold

            if not itemIsStatic then
                allItemsStatic = false
            end

            table.insert(items, {
                handle = handle,
                size = item.size,
                pose = {
                    pos[1], pos[2], pos[3],
                    quat[1], quat[2], quat[3], quat[4],
                },
                linearSpeed = linearSpeed,
                angularSpeed = angularSpeed,
                staticSteps = item.staticSteps,
                isStatic = itemIsStatic,
            })
        else
            allItemsStatic = false
        end
    end

    local capacity = math.max(1, segmentCount - 1)
    local full = #items == capacity and allItemsStatic

    sim.setInt32Signal('bufferFull', full and 1 or 0)

    sim.setStringSignal(
        'trackedItemsData',
        sim.packTable({
            segmentCount = segmentCount,
            capacity = capacity,
            full = full,
            allItemsStatic = allItemsStatic,
            items = items,
        })
    )
end


function initItemTrackingUi()
    trackedItems = {}
    trackedItemOrder = {}

    local xml = [[
    <ui title="Items on Conveyor" closeable="false" resizable="true" activate="false">
        <label id="101" text="Items: 0"/>
        <table id="100">
            <header>
                <item>Handle</item>
                <item>Dimension</item>
                <item>Pose [x y z qx qy qz qw]</item>
            </header>
            <row>
                <item>-</item>
                <item>-</item>
                <item>No items on conveyor</item>
            </row>
        </table>
    </ui>
    ]]

    itemUi = simUI.create(xml)

    simUI.setColumnWidth(itemUi, 100, 0, 70, 90)
    simUI.setColumnWidth(itemUi, 100, 1, 140, 180)
    simUI.setColumnWidth(itemUi, 100, 2, 430, 700)
end

function registerTrackedItem(handle, size)
    trackedItems[handle] = {
        size = {size[1], size[2], size[3]},
    }

    table.insert(trackedItemOrder, handle)
end

function objectPose(handle)
    local pos = sim.getObjectPosition(handle, sim.handle_world)
    local quat = sim.getObjectQuaternion(handle, sim.handle_world)
    return pos, quat
end

function isPoseOnConveyor(pos)
    local x0 = startPos[1]
    local x1 = startPos[1] + (segmentCount - 1) * segmentPitch

    local xmin = math.min(x0, x1) - xPad
    local xmax = math.max(x0, x1) + xPad

    return pos[1] >= xmin
        and pos[1] <= xmax
        and math.abs(pos[2] - startPos[2]) <= yPad
        and pos[3] >= zMin
        and pos[3] <= zMax
end

function pruneTrackedItems()
    local keptOrder = {}

    for i = 1, #trackedItemOrder do
        local handle = trackedItemOrder[i]
        local item = trackedItems[handle]

        if item ~= nil then
            local pos, quat = objectPose(handle)

            if pos ~= nil and isPoseOnConveyor(pos) then
                table.insert(keptOrder, handle)
            else
                trackedItems[handle] = nil
            end
        end
    end

    trackedItemOrder = keptOrder
end

function fmt3(v)
    return string.format('%.3f', v)
end

function formatDimension(size)
    return string.format(
        '[%s, %s, %s]',
        fmt3(size[1]),
        fmt3(size[2]),
        fmt3(size[3])
    )
end

function formatPose(pos, quat)
    return string.format(
        '[%s, %s, %s, %s, %s, %s, %s]',
        fmt3(pos[1]),
        fmt3(pos[2]),
        fmt3(pos[3]),
        fmt3(quat[1]),
        fmt3(quat[2]),
        fmt3(quat[3]),
        fmt3(quat[4])
    )
end

function updateItemTrackingUi()
    if not itemUi then
        return
    end

    pruneTrackedItems()

    local count = #trackedItemOrder
    simUI.setLabelText(itemUi, 101, 'Items: ' .. count, true)

    if count == 0 then
        simUI.setRowCount(itemUi, 100, 1, true)
        simUI.setItem(itemUi, 100, 0, 0, '-', true)
        simUI.setItem(itemUi, 100, 0, 1, '-', true)
        simUI.setItem(itemUi, 100, 0, 2, 'No items on conveyor', true)
        return
    end

    simUI.setRowCount(itemUi, 100, count, true)

    for row = 0, count - 1 do
        local handle = trackedItemOrder[row + 1]
        local item = trackedItems[handle]
        local pos, quat = objectPose(handle)

        simUI.setItem(itemUi, 100, row, 0, tostring(handle), true)
        simUI.setItem(itemUi, 100, row, 1, formatDimension(item.size), true)
        simUI.setItem(itemUi, 100, row, 2, formatPose(pos, quat), true)
    end
end

function sysCall_thread()
    currentTime = sim.getSimulationTime()

    while sim.getSimulationState() ~= sim.simulation_advancing_abouttostop do
        syncConfigFromSignals()
        updateItemTrackingUi()
        publishTrackedItemsData()

        local now = sim.getSimulationTime()

        if not isRobotPicking() and now - currentTime >= dropInterval then
            currentTime = now
            trySpawn()
        end

        sim.step()
    end
end

function sysCall_cleanup()
    if itemUi then
        simUI.destroy(itemUi)
    end
end
